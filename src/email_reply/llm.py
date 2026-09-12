"""Async LLM client with retry, JSON-mode, and a deterministic mock backend.

Supports:
- Gemini (default generator backend)
- OpenAI (default judge backend — deliberately a DIFFERENT model family
  from the generator to avoid self-preference bias in evaluation)
- Mock (deterministic, offline, for tests and CI)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def _env_key(name: str) -> str | None:
    v = os.environ.get(name)
    # allow sourcing from ~/.zshrc fallback (dev convenience only)
    if not v and name in ("GEMINI_API_KEY", "OPENAI_API_KEY"):
        try:
            with open(os.path.expanduser("~/.zshrc")) as f:
                for line in f:
                    m = re.match(rf'export {name}="([^"]+)"', line)
                    if m:
                        v = m.group(1)
                        break
        except OSError:
            pass
    return v


async def _post_json(url: str, payload: dict, headers: dict, timeout: int = 120) -> dict:
    loop = asyncio.get_event_loop()

    def do():
        req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                     headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:400]
            raise LLMError(f"HTTP {e.code}: {body}") from None
        except Exception as e:
            raise LLMError(str(e)) from None

    return await loop.run_in_executor(None, do)


async def gemini_generate(model: str, prompt: str, system: str | None = None,
                          temperature: float = 0.7, max_tokens: int = 2048,
                          json_mode: bool = False) -> str:
    key = _env_key("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    parts = [{"text": prompt}]
    contents = [{"role": "user", "parts": parts}]
    if system:
        payload_system = {"parts": [{"text": system}]}
    else:
        payload_system = None
    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            **({"responseMimeType": "application/json"} if json_mode else {}),
        },
    }
    if payload_system:
        body["systemInstruction"] = payload_system
    data = await _post_json(GEMINI_URL.format(model=model, key=key), body,
                            {"Content-Type": "application/json"})
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise LLMError(f"Gemini returned no text: {json.dumps(data)[:300]}") from None


async def openai_generate(model: str, prompt: str, system: str | None = None,
                          temperature: float = 0.7, max_tokens: int = 2048,
                          json_mode: bool = False) -> str:
    key = _env_key("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY not set")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": msgs,
        "temperature": temperature,
        **({"response_format": {"type": "json_object"}} if json_mode else {}),
    }
    data = await _post_json(OPENAI_URL, body, {"Content-Type": "application/json",
                                               "Authorization": f"Bearer {key}"})
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"OpenAI returned no content: {json.dumps(data)[:300]}") from None


# ------------------------------------------------------------------ mock

class MockLLM:
    """Deterministic offline 'LLM' for tests and CI.

    Produces plausible-but-boring text derived from the prompt so the whole
    pipeline (retrieval, generation, evaluation, reporting) runs with zero
    network access. The mock reply generator deliberately echoes required
    facts, so mock-mode scores are realistic (not floor-zero).
    """

    def __init__(self, style: str = "faithful"):
        self.style = style

    async def generate(self, model, prompt, system=None, temperature=0.7,
                       max_tokens=2048, json_mode=False) -> str:
        h = hashlib.sha256(prompt.encode()).hexdigest()[:8]
        if json_mode:
            # extract the JSON schema request from the prompt heuristically
            if '"claims"' in prompt or "claim" in prompt.lower():
                return json.dumps({"claims": ["the customer contacted support"]})
            return json.dumps({
                "scores": {"coverage": 0.8, "faithfulness": 0.9, "tone": 4.0,
                           "action_correctness": 0.85, "overall": 4.0},
                "rationale": f"mock judgment {h}",
            })
        # text mode: echo the instruction block back as an email
        return (f"[mock-{self.style}-{h}] Thank you for reaching out. "
                f"We have received your message and will follow up on the details "
                f"you provided. — The Northwind Team")


# ------------------------------------------------------------------ unified API

class _RateLimiter:
    """Token-bucket-ish global pacer: min interval between requests, in seconds."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last = 0.0

    async def acquire(self):
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._last + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# Global pacers per backend (free tiers are ~5 rpm; paid can go faster)
_PACERS: dict[str, _RateLimiter] = {}


def _pacer(backend: str) -> _RateLimiter:
    if backend not in _PACERS:
        env = os.environ.get("LLM_RPM", "")
        rpm = float(env) if env else 4.5  # conservative default for free tier
        _PACERS[backend] = _RateLimiter(60.0 / rpm)
    return _PACERS[backend]

class LLM:
    """Unified async wrapper with retry + exponential backoff + pacing."""

    _RATE_LIMIT_CODES = (429, 503)

    def __init__(self, backend: str = "gemini", model: str | None = None,
                 mock: MockLLM | None = None, rpm: float | None = None):
        self.backend = backend
        self.model = model or {"gemini": "gemini-3.5-flash",
                               "openai": "gpt-5.6-sol"}[backend]
        self.mock = mock
        if rpm is not None:
            _PACERS[backend] = _RateLimiter(60.0 / rpm)

    async def generate(self, prompt: str, system: str | None = None,
                       temperature: float = 0.7, max_tokens: int = 2048,
                       json_mode: bool = False, retries: int = 4) -> str:
        if self.backend == "mock":
            return await self.mock.generate(self.model, prompt, system, temperature,
                                            max_tokens, json_mode)
        pacer = _pacer(self.backend)
        last = None
        for attempt in range(retries):
            try:
                await pacer.acquire()
                if self.backend == "gemini":
                    return await gemini_generate(self.model, prompt, system, temperature,
                                                max_tokens, json_mode)
                return await openai_generate(self.model, prompt, system, temperature,
                                             max_tokens, json_mode)
            except LLMError as e:
                last = e
                msg = str(e)
                is_rate = any(f"HTTP {c}" in msg for c in self._RATE_LIMIT_CODES)
                if is_rate and attempt < retries - 1:
                    # rate limit: wait long (free-tier resets are minute-granular)
                    await asyncio.sleep(20 * (attempt + 1))
                    continue
                if "HTTP 5" in msg and attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        raise last


async def embed_gemini(texts: list[str], model: str = "gemini-embedding-001",
                       dims: int = 256) -> list[list[float]]:
    """Embed via Gemini API. Returns list of vectors (paced, batched 1-per-call)."""
    key = _env_key("GEMINI_API_KEY")
    if not key:
        raise LLMError("GEMINI_API_KEY not set")
    loop = asyncio.get_event_loop()
    out: list[list[float]] = []
    sem = asyncio.Semaphore(8)
    pacer = _pacer("gemini")

    async def one(t):
        async with sem:
            await pacer.acquire()

            def do():
                body = json.dumps({"model": f"models/{model}",
                                   "content": {"parts": [{"text": t[:8000]}]},
                                   "outputDimensionality": dims}).encode()
                req = urllib.request.Request(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent?key={key}",
                    data=body, headers={"Content-Type": "application/json"}, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=120) as r:
                        d = json.loads(r.read().decode())
                    e = d.get("embedding")
                    if isinstance(e, dict):
                        e = e.get("values")
                    return e
                except urllib.error.HTTPError as err:
                    raise LLMError(f"HTTP {err.code}: {err.read().decode()[:200]}") from None

            return await loop.run_in_executor(None, do)

    tasks = [one(t) for t in texts]
    out = await asyncio.gather(*tasks)
    return list(out)


async def embed_openai(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Embed via OpenAI. Returns list of embedding vectors."""
    key = _env_key("OPENAI_API_KEY")
    if not key:
        raise LLMError("OPENAI_API_KEY not set")
    out = []
    for i in range(0, len(texts), 512):
        chunk = texts[i:i + 512]
        body = json.dumps({"model": model, "input": chunk}).encode()
        req = urllib.request.Request("https://api.openai.com/v1/embeddings", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {key}"},
                                     method="POST")
        loop = asyncio.get_event_loop()

        def do(req=req):
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode())

        data = await loop.run_in_executor(None, do)
        out.extend([d["embedding"] for d in data["data"]])
    return out


def extract_json(text: str) -> dict:
    """Robustly extract the first JSON object from an LLM reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError(f"No JSON found in: {text[:200]}")
