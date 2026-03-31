# Student Submission Guidelines
## Vision Recognition Mini Project — Part 1

---

## 1. Overview

You are required to submit an **inference-ready folder** that the instructor's
automated evaluation script will use to test your models on a **hidden test set**.

The evaluation script will:
1. Import your `predictor.py`
2. Load your models using `load_classification_model()` and `load_detection_model()`
3. Run inference on hidden images using `predict_classification()` and `predict_detection_segmentation()`
4. Map your predictions to the canonical 5 classes using your class-mapping files
5. Compute final metrics

**You must not change any function name or signature.**

---

## 2. Required folder structure

Your submission folder must be named `VRMP1_<your_roll_number>` and contain:

```
VRMP1_<roll_number>/
├── predictor.py              ← Your filled-in inference code (includes class mappings)
├── validator_local.py        ← Local self-check script (do NOT modify)
├── requirements.txt          ← Python packages needed for YOUR inference
├── model_files/
│   ├── cls.pt (or cls.pth)   ← Classification model weights
│   └── seg.pt (or seg.pth)   ← Detection + segmentation model weights
```

---

## 3. The predictor.py file

Start from the provided `predictor.py` template. It contains **four functions**
you must implement. **Do NOT rename or remove any function.**

### Task 3.1 — Classification

```python
def load_classification_model(folder: str, device: str) -> Any:
    """Load your classification model from <folder>/model_files/cls.pt"""
    ...

def predict_classification(model, images: List[PIL.Image]) -> List[dict]:
    """
    Returns: [{"labels": [0, 1, 0, 1, 0]}, ...]
    - One dict per image
    - "labels" is a list of 5 binary ints (0 or 1), one per class
    - Order must match your CLS_CLASS_MAPPING dictionary
    """
    ...
```

### Task 3.2 — Detection + Segmentation

```python
def load_detection_model(folder: str, device: str) -> Any:
    """Load your detection+segmentation model from <folder>/model_files/seg.pt"""
    ...

def predict_detection_segmentation(model, images: List[PIL.Image]) -> List[dict]:
    """
    Returns: [{
        "boxes":  [[x1, y1, x2, y2], ...],    # float coords within image bounds
        "scores": [float, ...],                # confidence in [0, 1]
        "labels": [int, ...],                  # class indices per SEG_CLASS_MAPPING
        "masks":  [np.ndarray(H, W), ...]      # binary uint8 masks, same size as image
    }, ...]
    - One dict per image
    - boxes/scores/labels/masks must have the same length
    - Empty lists if no detections
    """
    ...
```

---

## 4. Class mapping dictionaries

You must fill in **two dictionaries** at the top of `predictor.py` that map
your model's output indices to the canonical category names.

### Canonical 5 categories

| Category Name       |
|---------------------|
| short sleeve top    |
| long sleeve top     |
| trousers            |
| shorts              |
| skirt               |

### CLS_CLASS_MAPPING

Maps your classification model's output indices → category names:

```python
CLS_CLASS_MAPPING = {
    0: "short sleeve top",
    1: "trousers",
    2: "shorts",
    3: "long sleeve top",
    4: "skirt",
}
```

Must have exactly **5 entries** (no background). Your indices can be in
**any order** — the evaluator uses the names to remap.

### SEG_CLASS_MAPPING

Maps your detection/segmentation model's output indices → category names.
If your model outputs a **background class** (common for Mask R-CNN, YOLO,
U-Net), include it — the evaluator will automatically ignore it:

```python
SEG_CLASS_MAPPING = {
    0: "background",
    1: "short sleeve top",
    2: "trousers",
    3: "shorts",
    4: "long sleeve top",
    5: "skirt",
}
```

**Important:**
- Category-name strings must match the canonical names **exactly** (case-insensitive).
- `CLS_CLASS_MAPPING` must have **5 entries** (one per class, no background).
- `SEG_CLASS_MAPPING` must have **5 or 6 entries** (5 clothing + optional background).
- The `"background"` entry will be skipped during evaluation.
- **Do NOT rename** either dictionary.

---

## 5. Evaluation metrics

Your models will be scored on the following:

| Task                     | Metric                                          |
|--------------------------|--------------------------------------------------|
| Classification           | **Macro F1-score** across all 5 classes           |
| Classification           | **Per-label macro accuracy** across all 5 classes |
| Detection                | **mAP @ [0.5 : 0.05 : 0.95]** (COCO-style)     |
| Segmentation             | **mIoU** (per-class IoU, macro-averaged)          |

---

## 6. requirements.txt

List only packages required for **your inference code** to run.
Do not include training-only dependencies.

Example:
```
torch
torchvision
ultralytics
numpy
Pillow
```

---

## 7. Important rules

1. **No internet access during evaluation.** All weights and dependencies must be
   in your folder. Do not download anything at runtime.

2. **Do not rename functions.** The evaluator imports your functions by exact name.

3. **Model size limit.** Models must have fewer than 7B parameters.

4. **Offline inference only.** No API calls, no external services.

5. **PIL images as input.** Your functions receive `PIL.Image.Image` objects
   (RGB mode). Handle them directly.

6. **Masks must be at original image resolution.** If your model internally
   resizes images (e.g., YOLO at 640×640), you **must resize the masks back**
   to the original input image dimensions before returning. The evaluator
   will attempt to auto-resize mismatched masks, but this may degrade
   your segmentation scores.

7. **Return exact formats.** Misformatted outputs will trigger assertion errors
   and score 0.

---

## 8. Self-validation

Before submitting, run the local validator to check your output format:

```bash
cd VRMP1_<your_roll_number>/
python validator_local.py
```

This checks:
- Your functions load and run without errors
- Classification outputs have the correct shape and binary values
- Detection outputs have matching-length boxes/scores/labels/masks
- Masks are binary and match image dimensions

The validator does **not** check metric quality — it only verifies format correctness.

---

## 9. What NOT to include

- `__pycache__/`
- `.venv/` or virtual environments
- Training datasets
- Experiment logs or `runs/` folders
- Notebook output caches
- OS-generated files (`__MACOSX`, `.DS_Store`)

Keep your folder clean and minimal.
