"""
YOLO Annotation Parser and Class Mapping for Facial Expression Dataset.
Validates, normalizes, and clamps bounding boxes to prevent image cropping errors.
"""

from typing import Tuple, List, Dict, Optional


# Authoritative class mapping directly from data.yaml
CLASS_NAMES: List[str] = [
    "angry",      # 0
    "contempt",   # 1
    "disgust",    # 2
    "fear",       # 3
    "happy",      # 4
    "natural",    # 5
    "sad",        # 6
    "sleepy",     # 7
    "surprised"   # 8
]

CLASS_NAME_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS_NAME: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def get_class_names() -> List[str]:
    """Returns the ordered list of facial expression classes."""
    return list(CLASS_NAMES)


def class_name_to_id(name: str) -> int:
    """Maps class name to integer ID."""
    name_clean = name.strip().lower()
    if name_clean not in CLASS_NAME_TO_ID:
        raise ValueError(f"Unknown class name: '{name}'. Expected one of {CLASS_NAMES}")
    return CLASS_NAME_TO_ID[name_clean]


def id_to_class_name(class_id: int) -> str:
    """Maps integer ID to class name."""
    if class_id not in ID_TO_CLASS_NAME:
        raise ValueError(f"Unknown class ID: {class_id}. Expected integer in [0, {len(CLASS_NAMES)-1}]")
    return ID_TO_CLASS_NAME[class_id]


def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Parses a single YOLO annotation line: '<class_id> <x_center> <y_center> <width> <height>'.
    Returns (class_id, xc, yc, w, h) or None if line is invalid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
        return class_id, xc, yc, w, h
    except (ValueError, TypeError):
        return None


def yolo_to_pixel_box(
    xc: float,
    yc: float,
    w: float,
    h: float,
    img_width: int,
    img_height: int,
    clamp: bool = True
) -> Tuple[int, int, int, int]:
    """
    Converts normalized YOLO coordinates (xc, yc, w, h) to pixel bounding box (xmin, ymin, xmax, ymax).
    Includes clamping to ensure bounding boxes stay strictly within image boundaries.
    """
    if clamp:
        # Clamp normalized values to [0.0, 1.0]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

    xmin = int(round((xc - w / 2.0) * img_width))
    ymin = int(round((yc - h / 2.0) * img_height))
    xmax = int(round((xc + w / 2.0) * img_width))
    ymax = int(round((yc + h / 2.0) * img_height))

    if clamp:
        xmin = max(0, min(img_width - 1, xmin))
        ymin = max(0, min(img_height - 1, ymin))
        xmax = max(xmin + 1, min(img_width, xmax))
        ymax = max(ymin + 1, min(img_height, ymax))

    return xmin, ymin, xmax, ymax
