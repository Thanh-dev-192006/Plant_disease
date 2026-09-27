"""Shared constants for all models — change here, not in notebooks."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "PlantVillage"
SPLITS_DIR = ROOT / "splits"
FIGURES_DIR = ROOT / "outputs" / "figures"
METRICS_DIR = ROOT / "outputs" / "metrics"
MODELS_DIR = ROOT / "models"

SEED = 42
IMG_SIZE = 224  # same input size for all 3 models (fair comparison)
BATCH_SIZE = 32
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# Files excluded after manual inspection (relative to RAW_DIR)
MANUAL_EXCLUDE = {
    # Screenshot of a diseased pepper fruit, not a healthy leaf -> wrong label
    "Pepper__bell___healthy/42f083e2-272d-4f83-ad9a-573ee90e50ec___Screen Shot 2015-05-06 at 4.01.13 PM.png",
}

# Near-duplicate = Pearson corr of 32x32 RGB thumbnails >= threshold (same class).
# 0.95 chosen by visual inspection: above it pairs are re-shots of the same leaf.
THUMB_SIZE = 32
NEAR_DUP_CORR = 0.95
