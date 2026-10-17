"""Shared constants that must stay identical across training and inference."""

CLASS_NAMES = ["Bleeding", "Ischemia", "Normal"]
CLASS_TO_IDX = {"Bleeding": 0, "Ischemia": 1, "Normal": 2}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

IMAGE_SIZE = 299
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

STROKE_CLASS_INDICES = (0, 1)
