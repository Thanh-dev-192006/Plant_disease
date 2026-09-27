"""Shared tf.data input pipeline — every model loads data through get_datasets().

Usage:
    from src.data import get_datasets, rescale_01
    train_ds, val_ds, test_ds = get_datasets(preprocess=rescale_01)
    # Model 3: preprocess=tf.keras.applications.mobilenet_v2.preprocess_input
"""
import json
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from src import config

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth() -> list:
    """Stop TF from grabbing all 6GB VRAM up front."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return gpus


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(config.SPLITS_DIR / f"{name}.csv")


def load_classes() -> list[dict]:
    return json.loads((config.SPLITS_DIR / "classes.json").read_text(encoding="utf-8"))


def class_names(pretty: bool = True) -> list[str]:
    return [c["name" if pretty else "dir"] for c in load_classes()]


def load_pixel_stats() -> dict:
    return json.loads((config.SPLITS_DIR / "pixel_stats.json").read_text(encoding="utf-8"))


def compute_class_weights(train_df: pd.DataFrame | None = None) -> dict[int, float]:
    """Balanced weights n / (k * n_c), for model.fit(class_weight=...)."""
    df = load_split("train") if train_df is None else train_df
    counts = df["class_idx"].value_counts().sort_index()
    n, k = counts.sum(), len(counts)
    return {int(c): float(n / (k * cnt)) for c, cnt in counts.items()}


# --- Preprocess functions: input float32 in [0, 255] ---

def rescale_01(x):
    return x / 255.0


def standardize(x):
    """(x/255 - mean) / std with RGB stats of the train set."""
    stats = load_pixel_stats()
    mean = tf.constant(stats["mean"], tf.float32)
    std = tf.constant(stats["std"], tf.float32)
    return (x / 255.0 - mean) / std


def build_augmentation(seed: int = config.SEED) -> tf.keras.Sequential:
    """Train-only augmentation; leaves have no canonical orientation."""
    L = tf.keras.layers
    return tf.keras.Sequential([
        L.RandomFlip("horizontal_and_vertical", seed=seed),
        L.RandomRotation(0.1, fill_mode="nearest", seed=seed),
        L.RandomZoom(0.1, fill_mode="nearest", seed=seed),
        L.RandomBrightness(0.1, value_range=(0, 255), seed=seed),
        L.RandomContrast(0.1, seed=seed),
    ], name="augmentation")


def _decode(path, label, img_size: int):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size), antialias=True)  # float32 [0,255]
    return img, label


def make_dataset(df: pd.DataFrame, img_size: int = config.IMG_SIZE,
                 batch_size: int = config.BATCH_SIZE, training: bool = False,
                 preprocess=None, augment: bool = True, cache=False,
                 seed: int = config.SEED) -> tf.data.Dataset:
    """cache: False | True (RAM) | path str (file cache on disk)."""
    paths = [str(config.RAW_DIR / p) for p in df["path"]]
    ds = tf.data.Dataset.from_tensor_slices((paths, df["class_idx"].values.astype("int32")))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: _decode(p, y, img_size), num_parallel_calls=AUTOTUNE)
    if cache:
        ds = ds.cache() if cache is True else ds.cache(str(cache))
    if training and augment:
        # Per-image (before batch): batched random layers fall back to slow while_loop in TF 2.10
        aug = build_augmentation(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if preprocess is not None:
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def get_datasets(img_size: int = config.IMG_SIZE, batch_size: int = config.BATCH_SIZE,
                 preprocess=None, augment: bool = True, cache=False):
    """Returns (train_ds, val_ds, test_ds). Only train is shuffled + augmented."""
    kw = dict(img_size=img_size, batch_size=batch_size, preprocess=preprocess)
    train = make_dataset(load_split("train"), training=True, augment=augment,
                         cache=_cache_for(cache, "train"), **kw)
    val = make_dataset(load_split("val"), cache=_cache_for(cache, "val"), **kw)
    test = make_dataset(load_split("test"), cache=_cache_for(cache, "test"), **kw)
    return train, val, test


def _cache_for(cache, name: str):
    """Directory cache -> one file per split."""
    if isinstance(cache, (str, os.PathLike)):
        os.makedirs(cache, exist_ok=True)
        return os.path.join(str(cache), f"{name}_cache")
    return cache
