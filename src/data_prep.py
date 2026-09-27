"""One-off dataset preparation: scan images, find duplicates, grouped stratified split."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config

def pretty_name(class_dir: str) -> str:
    """'Tomato__Tomato_YellowLeaf__Curl_Virus' -> 'Tomato YellowLeaf Curl Virus'."""
    crop, _, disease = class_dir.replace("Pepper__bell", "Pepper").partition("_")
    words = [w for w in disease.split("_") if w and w != crop]
    return f"{crop} {' '.join(words)}"


def crop_of(class_dir: str) -> str:
    return class_dir.split("_")[0]


def is_healthy(class_dir: str) -> bool:
    return class_dir.lower().endswith("healthy")


def _thumbnail(img: Image.Image, size: int = config.THUMB_SIZE) -> np.ndarray:
    """Tiny RGB thumbnail (uint8, flattened) used for near-duplicate search."""
    return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.uint8).ravel()


def _scan_one(path: Path) -> dict:
    rel = path.relative_to(config.RAW_DIR).as_posix()
    row = {
        "path": rel,
        "class_dir": path.parent.name,
        "ext": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "width": None, "height": None, "mode": None,
        "md5": None, "thumb": None, "readable": False, "error": None,
    }
    try:
        data = path.read_bytes()
        row["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(path) as img:
            img.load()  # full decode, catches truncated files
            row.update(width=img.width, height=img.height, mode=img.mode,
                       thumb=_thumbnail(img), readable=True)
    except Exception as exc:  # corrupt / empty / non-image
        row["error"] = f"{type(exc).__name__}: {exc}"[:120]
    return row


def scan_images(raw_dir: Path = config.RAW_DIR, workers: int = 16) -> pd.DataFrame:
    """Decode every file once; returns one row per file with metadata + hashes."""
    files = sorted(p for p in raw_dir.glob("*/*") if p.is_file())
    with ThreadPoolExecutor(workers) as pool:
        rows = list(pool.map(_scan_one, files))
    return pd.DataFrame(rows)


def exclusion_reason(row) -> str | None:
    if row["path"] in config.MANUAL_EXCLUDE:
        return "manual: wrong label"
    if not row["readable"]:
        return "unreadable"
    if row["ext"] not in config.IMAGE_EXTS:
        return "not an image extension"
    if row["mode"] != "RGB":
        return f"mode {row['mode']}"
    return None


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n)

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)


def near_duplicate_pairs(thumbs: np.ndarray, min_corr: float) -> list[tuple[int, int, float]]:
    """All (i, j, r) with i<j and Pearson corr of thumbnails >= min_corr."""
    x = thumbs.astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    sim = x @ x.T
    iu, ju = np.triu_indices(len(x), k=1)
    r = sim[iu, ju]
    keep = r >= min_corr
    return list(zip(iu[keep].tolist(), ju[keep].tolist(), r[keep].round(4).tolist()))


def assign_groups(df: pd.DataFrame, min_corr: float = config.NEAR_DUP_CORR):
    """Group exact dups (any class) + near dups (same class). Returns (group_ids, near_pairs_df)."""
    df = df.reset_index(drop=True)
    uf = _UnionFind(len(df))
    for idx in df.groupby("md5").indices.values():
        for j in idx[1:]:
            uf.union(idx[0], j)

    near = []
    for _, idx in df.groupby("class_dir").indices.items():
        thumbs = np.stack(df.loc[idx, "thumb"].to_numpy())
        for a, b, r in near_duplicate_pairs(thumbs, min_corr):
            uf.union(idx[a], idx[b])
            near.append((df.at[idx[a], "path"], df.at[idx[b], "path"], r))

    groups = np.array([uf.find(i) for i in range(len(df))])
    near_df = pd.DataFrame(near, columns=["path_a", "path_b", "corr"])
    return groups, near_df


def grouped_stratified_split(df: pd.DataFrame, ratios=config.SPLIT_RATIOS,
                             seed: int = config.SEED) -> pd.Series:
    """Per class: shuffle groups, give each to the split furthest below its target size."""
    rng = np.random.default_rng(seed)
    names = ["train", "val", "test"]
    split = pd.Series(index=df.index, dtype=object)
    for _, cdf in df.groupby("class_dir"):
        group_ids = cdf["group"].unique()
        rng.shuffle(group_ids)
        targets = np.array(ratios) * len(cdf)
        counts = np.zeros(3)
        for g in group_ids:
            members = cdf.index[cdf["group"] == g]
            k = int(np.argmax(targets - counts))
            split[members] = names[k]
            counts[k] += len(members)
    return split


def save_splits(df: pd.DataFrame, out_dir: Path = config.SPLITS_DIR) -> list[str]:
    """Write train/val/test CSVs + classes.json. Returns ordered class list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(df["class_dir"].unique())
    idx = {c: i for i, c in enumerate(classes)}
    for name in ["train", "val", "test"]:
        part = df[df["split"] == name].sort_values("path")
        pd.DataFrame({
            "path": part["path"],
            "label": part["class_dir"],
            "class_idx": part["class_dir"].map(idx),
        }).to_csv(out_dir / f"{name}.csv", index=False)
    meta = [{"idx": i, "dir": c, "name": pretty_name(c)} for i, c in enumerate(classes)]
    (out_dir / "classes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return classes


def _channel_sums(path: Path, size: int):
    with Image.open(path) as img:
        x = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), np.float64) / 255.0
    return x.sum((0, 1)), (x ** 2).sum((0, 1)), x.shape[0] * x.shape[1]


def pixel_stats(paths, size: int = config.IMG_SIZE, workers: int = 16) -> dict:
    """Per-channel RGB mean/std in [0,1], computed at training resolution."""
    with ThreadPoolExecutor(workers) as pool:
        res = list(pool.map(lambda p: _channel_sums(config.RAW_DIR / p, size), paths))
    s = sum(r[0] for r in res)
    s2 = sum(r[1] for r in res)
    n = sum(r[2] for r in res)
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return {"mean": mean.round(4).tolist(), "std": std.round(4).tolist(), "n_images": len(res)}
