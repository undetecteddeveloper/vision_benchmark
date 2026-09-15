"""
Step 2 - Send every sampled image to one model with the identical detection
prompt (see common.py) and save its predictions in COCO results format.

Requires an API key as an environment variable:
    DeepSeek : DEEPSEEK_API_KEY   (get one at platform.deepseek.com)
    Gemini   : GEMINI_API_KEY     (get one at aistudio.google.com)

Usage:
    python query_models.py --model deepseek
    python query_models.py --model gemini

Run both before evaluate.py. Each run writes data/predictions_<model>.json
and data/failures_<model>.json (images that errored or produced no
parseable boxes, so you can see failure rate as its own signal, not just
mAP - a model that silently skips hard images will look artificially good
on mAP alone).
"""
import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import build_prompt, extract_json_array, detections_to_coco_results

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"


def get_image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (width, height)


def encode_image_b64(path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def call_deepseek(image_path, width, height, api_key):
    prompt = build_prompt(width, height)
    b64 = encode_image_b64(image_path)
    body = {
        "model": "deepseek-flash",  # DeepSeek's current alias for V4.1 Flash
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def call_gemini(image_path, width, height, api_key):
    prompt = build_prompt(width, height)
    b64 = encode_image_b64(image_path)
    body = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            ],
        }],
        "generationConfig": {"temperature": 0},
    }
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent",
        data=json.dumps(body).encode(),
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["candidates"][0]["content"]["parts"][0]["text"]


CALLERS = {"deepseek": call_deepseek, "gemini": call_gemini}
KEY_ENV_VARS = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(CALLERS))
    parser.add_argument("--sleep", type=float, default=1.0,
                         help="Seconds to sleep between requests (rate-limit courtesy)")
    args = parser.parse_args()

    api_key = os.environ.get(KEY_ENV_VARS[args.model])
    if not api_key:
        sys.exit(f"Set {KEY_ENV_VARS[args.model]} in your environment first.")

    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        sys.exit("Run setup_data.py first - data/ground_truth.json is missing.")
    ground_truth = json.loads(gt_path.read_text())
    images = ground_truth["images"]

    caller = CALLERS[args.model]
    all_results = []
    failures = []

    for i, img in enumerate(images, 1):
        path = IMAGES_DIR / img["file_name"]
        if not path.exists():
            failures.append({"image_id": img["id"], "reason": "image file missing"})
            continue
        width, height = get_image_size(path)
        try:
            raw_text = caller(path, width, height, api_key)
            detections = extract_json_array(raw_text)
            results, dropped = detections_to_coco_results(detections, img["id"], width, height)
            all_results.extend(results)
            if dropped:
                failures.append({
                    "image_id": img["id"],
                    "reason": "some detections dropped (bad label or box)",
                    "dropped": dropped,
                })
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError) as e:
            failures.append({"image_id": img["id"], "reason": str(e)})

        print(f"[{args.model}] {i}/{len(images)} - "
              f"{len(all_results)} boxes so far, {len(failures)} failures")
        time.sleep(args.sleep)

    out_path = DATA_DIR / f"predictions_{args.model}.json"
    out_path.write_text(json.dumps(all_results))
    fail_path = DATA_DIR / f"failures_{args.model}.json"
    fail_path.write_text(json.dumps(failures, indent=2))

    print(f"\nWrote {out_path} ({len(all_results)} predicted boxes across {len(images)} images)")
    print(f"Wrote {fail_path} ({len(failures)} images had an error or a dropped detection)")


if __name__ == "__main__":
    main()
