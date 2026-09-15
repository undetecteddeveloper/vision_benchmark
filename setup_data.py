"""
Step 1 - Build a small, labeled multi-object-detection test set from COCO val2017.

One-time manual step first (COCO doesn't let scripts fetch the annotation
file from an anonymous, rate-limited context, so grab it yourself):

    Download: https://images.cocodataset.org/annotations/annotations_trainval2017.zip  (~241 MB)
    Unzip it, then copy annotations/instances_val2017.json next to this script.

Then run:

    python setup_data.py --n 50 --min-objects 3 --seed 42

This will:
  1. Randomly sample N images from val2017 that have at least
     --min-objects annotated instances (so the test is actually about
     *multi*-object detection, not single-object images).
  2. Download just those N images (not the full 1GB val2017.zip) from the
     COCO CDN, since each image is individually addressable.
  3. Write data/ground_truth.json - a COCO-format file containing only the
     sampled images/annotations/categories, for use by evaluate.py.
"""
import argparse
import json
import random
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
IMAGES_DIR = DATA_DIR / "images"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=str(HERE / "instances_val2017.json"),
                         help="Path to COCO's instances_val2017.json")
    parser.add_argument("--n", type=int, default=50, help="Number of images to sample")
    parser.add_argument("--min-objects", type=int, default=3,
                         help="Only sample images with at least this many annotated instances")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        sys.exit(
            f"Can't find {ann_path}.\n"
            "Download https://images.cocodataset.org/annotations/annotations_trainval2017.zip, "
            "unzip it, and copy instances_val2017.json next to this script first."
        )

    print(f"Loading {ann_path} ...")
    coco = json.loads(ann_path.read_text())

    anns_by_image = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    eligible_image_ids = [
        img_id for img_id, anns in anns_by_image.items()
        if len(anns) >= args.min_objects
    ]
    print(f"{len(eligible_image_ids)} / {len(anns_by_image)} images have "
          f">= {args.min_objects} annotated objects.")

    random.seed(args.seed)
    if len(eligible_image_ids) < args.n:
        sys.exit(f"Only {len(eligible_image_ids)} eligible images, but --n={args.n} requested.")
    sampled_ids = set(random.sample(eligible_image_ids, args.n))

    images_by_id = {img["id"]: img for img in coco["images"]}
    sampled_images = [images_by_id[i] for i in sampled_ids]
    sampled_anns = [a for i in sampled_ids for a in anns_by_image[i]]

    ground_truth = {
        "images": sampled_images,
        "annotations": sampled_anns,
        "categories": coco["categories"],
    }

    DATA_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    (DATA_DIR / "ground_truth.json").write_text(json.dumps(ground_truth))
    print(f"Wrote {DATA_DIR / 'ground_truth.json'} "
          f"({len(sampled_images)} images, {len(sampled_anns)} boxes).")

    print("Downloading images ...")
    for i, img in enumerate(sampled_images, 1):
        dest = IMAGES_DIR / img["file_name"]
        if dest.exists():
            continue
        url = img["coco_url"]
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as e:
            print(f"  ! failed to download {url}: {e}")
        if i % 10 == 0 or i == len(sampled_images):
            print(f"  {i}/{len(sampled_images)}")

    print("Done. Images are in data/images/, ground truth is data/ground_truth.json")


if __name__ == "__main__":
    main()
