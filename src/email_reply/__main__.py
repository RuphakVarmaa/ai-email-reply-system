"""Entry point for `python -m email_reply`."""
import asyncio
import sys


def main():
    # Forward to pipeline module
    from .pipeline import main as pipeline_main
    asyncio.run(pipeline_main())


if __name__ == "__main__":
    main()
