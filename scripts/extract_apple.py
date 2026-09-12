"""Extract AppleSupport conversation pairs from the TWCS dataset.

Usage:
    python scripts/extract_apple.py path/to/twcs.csv data/raw/apple_pairs.jsonl
"""
import csv
import json
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_apple.py <twcs.csv> <output.jsonl>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    out_path = sys.argv[2]
    brand = "AppleSupport"
    
    print(f"Loading {csv_path}...", flush=True)
    rows = {}
    with open(csv_path, errors="replace") as f:
        for row in csv.DictReader(f):
            rows[row["tweet_id"]] = row
    print(f"Total rows: {len(rows)}")
    
    brand_tweets = {tid: r for tid, r in rows.items() if r["author_id"] == brand}
    customer_to_brand = {}
    for tid, r in brand_tweets.items():
        ref = r.get("in_response_to_tweet_id", "")
        if ref and ref in rows and rows[ref]["inbound"] == "True":
            customer_to_brand[ref] = tid
    
    pairs = []
    for cust_tid, brand_tid in customer_to_brand.items():
        cust = rows[cust_tid]
        reply = rows[brand_tid]
        pairs.append({
            "customer_tweet_id": cust_tid,
            "brand_tweet_id": brand_tid,
            "customer_author": cust["author_id"],
            "customer_text": cust["text"],
            "brand_text": reply["text"],
            "customer_time": cust.get("created_at", ""),
            "brand_time": reply.get("created_at", ""),
        })
    
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    
    print(f"Extracted {len(pairs)} AppleSupport pairs -> {out_path}")


if __name__ == "__main__":
    main()
