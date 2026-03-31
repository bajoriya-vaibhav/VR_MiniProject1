"""
predictor.py — Student inference file for hidden evaluation.

╔══════════════════════════════════════════════════════════════════╗
║  DO NOT RENAME ANY FUNCTION.                                    ║
║  DO NOT CHANGE FUNCTION SIGNATURES.                             ║
║  DO NOT REMOVE ANY FUNCTION.                                    ║
║  DO NOT RENAME CLS_CLASS_MAPPING or SEG_CLASS_MAPPING.          ║
║  You may add helper functions / imports as needed.              ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# CLASS MAPPINGS
# Matches label_map from training:
#   TOP5 = ['short sleeve top', 'trousers', 'shorts', 'long sleeve top', 'skirt']
# ═══════════════════════════════════════════════════════════════════

CLS_CLASS_MAPPING: Dict[int, str] = {
    0: "short sleeve top",
    1: "trousers",
    2: "shorts",
    3: "long sleeve top",
    4: "skirt",
}

# Mask R-CNN uses 0=background, 1-5=clothing (same order as label_map)
SEG_CLASS_MAPPING: Dict[int, str] = {
    0: "background",
    1: "short sleeve top",
    2: "trousers",
    3: "shorts",
    4: "long sleeve top",
    5: "skirt",
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _find_weights(folder: Path, stem: str) -> Path:
    for ext in (".pt", ".pth"):
        candidate = folder / "model_files" / (stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No weights file found for '{stem}' in {folder / 'model_files'}"
    )


# Classification preprocessing — must match training (Resize 160x160, ToTensor)
_CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# Mask R-CNN input size used during training
_SEG_SIZE = 416


# ═══════════════════════════════════════════════════════════════════
# TASK 3.1 — CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

def load_classification_model(folder: str, device: str) -> Any:
    import timm

    weights_path = _find_weights(Path(folder), "cls")

    # ResNet50 — matches the weights in cls.pth
    model = timm.create_model("resnet50", pretrained=False, num_classes=5)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return {"model": model, "device": device}

def predict_classification(model: Any, images: List[Image.Image]) -> List[Dict]:
    """
    Multi-label classification: returns binary label vector per image.
    Output order matches CLS_CLASS_MAPPING (indices 0-4).
    """
    net    = model["model"]
    device = model["device"]

    results = []
    with torch.no_grad():
        for img in images:
            img_rgb = img.convert("RGB")
            tensor  = _CLS_TRANSFORM(img_rgb).unsqueeze(0).to(device)
            logits  = net(tensor)                          # (1, 5)
            probs   = torch.sigmoid(logits).squeeze(0)    # (5,)
            preds   = (probs > 0.5).int().cpu().tolist()  # [0/1 x 5]
            results.append({"labels": preds})

    return results


# ═══════════════════════════════════════════════════════════════════
# TASK 3.2 — DETECTION + INSTANCE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════

def load_detection_model(folder: str, device: str) -> Any:
    """
    Load Mask R-CNN (ResNet50-FPN) fine-tuned for 5 clothing classes.
    Weights: model_files/seg.pth
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor

    NUM_CLASSES = 5  # excluding background

    weights_path = _find_weights(Path(folder), "seg")

    model = maskrcnn_resnet50_fpn(weights=None)
    in_f  = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor  = FastRCNNPredictor(in_f, NUM_CLASSES + 1)
    in_m  = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, NUM_CLASSES + 1)

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return {"model": model, "device": device}


def predict_detection_segmentation(
    model: Any,
    images: List[Image.Image],
) -> List[Dict]:
    """
    Instance detection + segmentation using Mask R-CNN.
    Masks are resized back to original image resolution before returning.
    Labels match SEG_CLASS_MAPPING (1-5, background=0 excluded from output).
    """
    net    = model["model"]
    device = model["device"]

    results = []

    with torch.no_grad():
        for img in images:
            img_rgb  = img.convert("RGB")
            orig_w, orig_h = img_rgb.size          # original resolution

            # resize to training size
            img_resized = img_rgb.resize((_SEG_SIZE, _SEG_SIZE))
            tensor      = transforms.functional.to_tensor(img_resized).unsqueeze(0).to(device)

            preds = net(tensor)[0]

            boxes_out  = []
            scores_out = []
            labels_out = []
            masks_out  = []

            scale_x = orig_w / _SEG_SIZE
            scale_y = orig_h / _SEG_SIZE

            for i in range(len(preds["scores"])):
                score = float(preds["scores"][i].cpu())
                if score < 0.3:          # confidence threshold
                    continue

                label = int(preds["labels"][i].cpu())
                if label == 0:           # skip background
                    continue

                # scale box back to original resolution
                x1, y1, x2, y2 = preds["boxes"][i].cpu().tolist()
                x1 = max(0.0, x1 * scale_x)
                y1 = max(0.0, y1 * scale_y)
                x2 = min(float(orig_w), x2 * scale_x)
                y2 = min(float(orig_h), y2 * scale_y)

                if x2 <= x1 or y2 <= y1:
                    continue

                # threshold float mask → binary uint8 at original resolution
                mask_float = preds["masks"][i].squeeze(0).cpu().numpy()  # (H_seg, W_seg)
                mask_bin   = (mask_float > 0.5).astype(np.uint8)
                mask_pil   = Image.fromarray(mask_bin * 255).resize(
                    (orig_w, orig_h), Image.NEAREST
                )
                mask_final = (np.array(mask_pil) > 127).astype(np.uint8)

                boxes_out.append([x1, y1, x2, y2])
                scores_out.append(score)
                labels_out.append(label)
                masks_out.append(mask_final)

            results.append({
                "boxes":  boxes_out,
                "scores": scores_out,
                "labels": labels_out,
                "masks":  masks_out,
            })

    return results
