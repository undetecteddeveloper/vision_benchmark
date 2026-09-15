"""
Shared constants and helpers for the bbox evaluation harness.
Both query scripts and the evaluator import from here so the prompt,
category list, and output schema stay identical for every model tested.
"""
import json
import re

# The 80 COCO "thing" categories, in COCO's official category-id order.
# Keeping (id, name) together means predictions can be mapped straight
# back to COCO category_ids for pycocotools without a second lookup table.
COCO_CATEGORIES = [
    (1, "person"), (2, "bicycle"), (3, "car"), (4, "motorcycle"), (5, "airplane"),
    (6, "bus"), (7, "train"), (8, "truck"), (9, "boat"), (10, "traffic light"),
    (11, "fire hydrant"), (13, "stop sign"), (14, "parking meter"), (15, "bench"),
    (16, "bird"), (17, "cat"), (18, "dog"), (19, "horse"), (20, "sheep"),
    (21, "cow"), (22, "elephant"), (23, "bear"), (24, "zebra"), (25, "giraffe"),
    (27, "backpack"), (28, "umbrella"), (31, "handbag"), (32, "tie"),
    (33, "suitcase"), (34, "frisbee"), (35, "skis"), (36, "snowboard"),
    (37, "sports ball"), (38, "kite"), (39, "baseball bat"), (40, "baseball glove"),
    (41, "skateboard"), (42, "surfboard"), (43, "tennis racket"), (44, "bottle"),
    (46, "wine glass"), (47, "cup"), (48, "fork"), (49, "knife"), (50, "spoon"),
    (51, "bowl"), (52, "banana"), (53, "apple"), (54, "sandwich"), (55, "orange"),
    (56, "broccoli"), (57, "carrot"), (58, "hot dog"), (59, "pizza"), (60, "donut"),
    (61, "cake"), (62, "chair"), (63, "couch"), (64, "potted plant"), (65, "bed"),
    (67, "dining table"), (70, "toilet"), (72, "tv"), (73, "laptop"), (74, "mouse"),
    (75, "remote"), (76, "keyboard"), (77, "cell phone"), (78, "microwave"),
    (79, "oven"), (80, "toaster"), (81, "sink"), (82, "refrigerator"), (84, "book"),
    (85, "clock"), (86, "vase"), (87, "scissors"), (88, "teddy bear"),
    (89, "hair drier"), (90, "toothbrush"),
]
NAME_TO_ID = {name: cid for cid, name in COCO_CATEGORIES}

PROMPT_TEMPLATE = """You are an object detector. The image is exactly {width}x{height} pixels.

Detect every instance of the following object categories that appears in the image:
{category_list}

Return ONLY a JSON array and nothing else - no markdown code fences, no commentary.
Each element must be an object with exactly these two keys:
  "label": one of the category names above, copied exactly as written
  "box_2d": [x_min, y_min, x_max, y_max] in pixel coordinates of the ORIGINAL
            {width}x{height} image, with (0, 0) at the top-left corner

Include one entry per object instance (so a category can appear more than once).
If none of the listed categories appear in the image, return [].
"""


def build_prompt(width: int, height: int) -> str:
    names = ", ".join(name for _, name in COCO_CATEGORIES)
    return PROMPT_TEMPLATE.format(width=width, height=height, category_list=names)


def extract_json_array(raw_text: str):
    """
    Models occasionally wrap the JSON in ```json ... ``` fences or add a
    stray sentence before/after it despite instructions. This pulls out
    the first top-level [...] array and parses it, raising a clear error
    (rather than silently returning []) if nothing parseable is found.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in model output:\n{raw_text[:500]}")
    return json.loads(match.group(0))


def detections_to_coco_results(detections, image_id, width, height):
    """
    Convert the model's [{"label":..., "box_2d":[xmin,ymin,xmax,ymax]}, ...]
    into COCO's results format: [{"image_id","category_id","bbox":[x,y,w,h],"score"}, ...]
    Boxes are clamped to the image bounds and any entry with an unrecognized
    label or a degenerate (zero/negative area) box is dropped rather than
    crashing the whole run - malformed entries are logged by the caller.
    """
    results = []
    dropped = []
    for det in detections:
        label = det.get("label")
        box = det.get("box_2d")
        cat_id = NAME_TO_ID.get(label)
        if cat_id is None or not (isinstance(box, list) and len(box) == 4):
            dropped.append(det)
            continue
        x_min, y_min, x_max, y_max = box
        x_min = max(0, min(x_min, width))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height))
        y_max = max(0, min(y_max, height))
        w, h = x_max - x_min, y_max - y_min
        if w <= 0 or h <= 0:
            dropped.append(det)
            continue
        results.append({
            "image_id": image_id,
            "category_id": cat_id,
            "bbox": [x_min, y_min, w, h],
            # Neither model is asked for a confidence score, and COCOeval
            # requires one - a constant score means every detection is
            # weighted equally, which is fine since we aren't ranking by
            # confidence, only checking whether the box is right.
            "score": 1.0,
        })
    return results, dropped
