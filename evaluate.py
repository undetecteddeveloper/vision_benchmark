"""
Step 3 - Score both models' predictions against the same ground truth using
pycocotools, the reference implementation behind the mAP numbers reported
by COCO-style leaderboards (including Roboflow's Vision Evals).

Usage:
    python evaluate.py
"""
import contextlib
import io
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

METRIC_NAMES = [
    "mAP @[.50:.95]", "mAP @.50", "mAP @.75",
    "mAP (small)", "mAP (medium)", "mAP (large)",
    "AR @1", "AR @10", "AR @100",
    "AR (small)", "AR (medium)", "AR (large)",
]


def evaluate_one(gt_path, pred_path):
    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools' own prints
        coco_gt = COCO(str(gt_path))
        preds = json.loads(pred_path.read_text())
        if not preds:
            return None
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev.stats  # 12 floats, order matches METRIC_NAMES


def main():
    gt_path = DATA_DIR / "ground_truth.json"
    if not gt_path.exists():
        raise SystemExit("Run setup_data.py first.")

    models = {}
    for model in ["deepseek", "gemini"]:
        pred_path = DATA_DIR / f"predictions_{model}.json"
        if pred_path.exists():
            stats = evaluate_one(gt_path, pred_path)
            if stats is None:
                print(f"{model}: predictions file is empty (0 boxes) - skipping")
            else:
                models[model] = stats
        else:
            print(f"{model}: no predictions_{model}.json found - run "
                  f"query_models.py --model {model} first")

    if not models:
        return

    name_col_width = max(len(n) for n in METRIC_NAMES)
    header = f"{'Metric':<{name_col_width}}  " + "  ".join(f"{m:>10}" for m in models)
    print("\n" + header)
    print("-" * len(header))
    for i, name in enumerate(METRIC_NAMES):
        row = f"{name:<{name_col_width}}  " + "  ".join(
            f"{stats[i]:>10.3f}" for stats in models.values()
        )
        print(row)

    print(
        "\nmAP @.50 is the metric quoted as 'mAP@50' in most detection "
        "leaderboards (Roboflow's Vision Evals included) - that's the row "
        "most directly comparable to the numbers discussed earlier."
    )


if __name__ == "__main__":
    main()
