"""Patch dataset for U-Net training, built on top of rasterize.py's output.

Each "scene" is a directory produced by `python ml/rasterize.py --out <scene_dir> ...`,
containing intensity.tif, height.tif, mask.tif and meta.json. This module tiles those
full-scene rasters into fixed-size patches, drops patches with no LiDAR coverage at
all, downsamples empty-of-label ("background-only") patches so the severe class
imbalance (background is ~98%+ of labeled pixels - see project notes) doesn't drown
out lane/crosswalk/stop-line signal, and normalizes each channel per-scene.

Nodata handling: intensity.tif/height.tif use -9999 where the point cloud has no
returns (common - LAS coverage is a driving swath, not a full rectangle; ~20% valid
was typical in testing). Those pixels carry no signal, so they're excluded from the
loss via IGNORE_INDEX in the target rather than being trained as "background".
"""
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data import Dataset

NODATA = -9999
IGNORE_INDEX = 255


@dataclass
class Scene:
    name: str
    intensity: np.ndarray
    height: np.ndarray
    mask: np.ndarray
    valid: np.ndarray  # bool, True where intensity/height are real LiDAR returns
    intensity_mean: float = 0.0
    intensity_std: float = 1.0
    height_mean: float = 0.0
    height_std: float = 1.0


def load_scene(scene_dir: Path, require_mask: bool = True) -> Scene:
    """require_mask=False is for inference on a brand-new area that has no
    digitized ground-truth SHP yet (so rasterize.py was run --las-only, no
    mask.tif) - train.py/build_patch_index need real labels so always require
    it there; infer.py never reads scene.mask, so it can skip this."""
    intensity = tifffile.imread(scene_dir / "intensity.tif").astype(np.float32)
    height = tifffile.imread(scene_dir / "height.tif").astype(np.float32)
    mask_path = scene_dir / "mask.tif"
    if mask_path.exists():
        mask = tifffile.imread(mask_path).astype(np.uint8)
        if intensity.shape != mask.shape or height.shape != mask.shape:
            raise ValueError(f"{scene_dir}: intensity/height/mask shape mismatch")
    elif require_mask:
        raise FileNotFoundError(f"{mask_path} not found - training needs real ground-truth labels")
    else:
        mask = np.zeros(intensity.shape, dtype=np.uint8)

    valid = (intensity != NODATA) & (height != NODATA)
    scene = Scene(name=scene_dir.name, intensity=intensity, height=height, mask=mask, valid=valid)
    if valid.any():
        scene.intensity_mean = float(intensity[valid].mean())
        scene.intensity_std = float(intensity[valid].std()) or 1.0
        scene.height_mean = float(height[valid].mean())
        scene.height_std = float(height[valid].std()) or 1.0
    return scene


@dataclass
class PatchIndex:
    scene_idx: int
    y: int
    x: int
    has_label: bool


def _tile_positions(size: int, patch: int, stride: int):
    if size <= patch:
        return [0]
    positions = list(range(0, size - patch + 1, stride))
    if positions[-1] != size - patch:
        positions.append(size - patch)
    return positions


def build_patch_index(scenes: list[Scene], patch_size: int, stride: int,
                       min_valid_frac: float, bg_keep_ratio: float, seed: int = 0) -> list[PatchIndex]:
    """Scans every scene on a (patch_size, stride) grid and keeps:
    - every patch touching at least one labeled (non-background) pixel
    - a random bg_keep_ratio fraction of otherwise-valid background-only patches
    Patches with fewer than min_valid_frac of pixels covered by real LiDAR returns
    are dropped outright (nothing to learn from).
    """
    rng = random.Random(seed)
    positive, background = [], []
    for scene_idx, scene in enumerate(scenes):
        h, w = scene.mask.shape
        for y in _tile_positions(h, patch_size, stride):
            for x in _tile_positions(w, patch_size, stride):
                valid_patch = scene.valid[y:y + patch_size, x:x + patch_size]
                valid_frac = valid_patch.mean() if valid_patch.size else 0.0
                if valid_frac < min_valid_frac:
                    continue
                mask_patch = scene.mask[y:y + patch_size, x:x + patch_size]
                has_label = bool(((mask_patch != 0) & valid_patch).any())
                entry = PatchIndex(scene_idx, y, x, has_label)
                (positive if has_label else background).append(entry)

    n_keep_bg = round(len(background) * bg_keep_ratio)
    background = rng.sample(background, min(n_keep_bg, len(background)))
    patches = positive + background
    rng.shuffle(patches)
    return patches


class PatchDataset(Dataset):
    def __init__(self, scenes: list[Scene], patches: list[PatchIndex], patch_size: int, augment: bool = False):
        self.scenes = scenes
        self.patches = patches
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        p = self.patches[idx]
        scene = self.scenes[p.scene_idx]
        s = self.patch_size
        sl = (slice(p.y, p.y + s), slice(p.x, p.x + s))

        intensity = (scene.intensity[sl] - scene.intensity_mean) / scene.intensity_std
        height = (scene.height[sl] - scene.height_mean) / scene.height_std
        valid = scene.valid[sl]
        intensity[~valid] = 0.0
        height[~valid] = 0.0

        target = scene.mask[sl].astype(np.int64)
        target[~valid] = IGNORE_INDEX

        x = np.stack([intensity, height], axis=0)

        if self.augment:
            if random.random() < 0.5:
                x = x[:, :, ::-1]
                target = target[:, ::-1]
            if random.random() < 0.5:
                x = x[:, ::-1, :]
                target = target[::-1, :]
            k = random.randint(0, 3)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
                target = np.rot90(target, k, axes=(0, 1))

        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(target))


def load_scenes(scene_dirs: list[Path]) -> list[Scene]:
    return [load_scene(d) for d in scene_dirs]


def class_weights(scenes: list[Scene], num_classes: int, power: float = 0.5,
                   max_weight: float | None = 20.0) -> torch.Tensor:
    """Inverse-frequency**power weights over labeled (valid, non-ignored) pixels, for CrossEntropyLoss.

    Raw inverse-frequency (power=1) gave stop_line/other weights in the
    100s-1000s range - confirmed on real data (v2 checkpoint) that this makes
    the model spam rare-class false positives broadly (speckle noise off-road,
    lane_line covering the whole road surface) rather than localizing them,
    since CE barely penalizes false positives relative to the huge reward for
    catching a rare-class pixel. sqrt (power=0.5) plus a hard cap compresses
    that dynamic range so rare classes still get a training boost without the
    loss being dominated by them.
    """
    counts = np.zeros(num_classes, dtype=np.float64)
    for scene in scenes:
        vals, cnts = np.unique(scene.mask[scene.valid], return_counts=True)
        for v, c in zip(vals, cnts):
            if v < num_classes:
                counts[v] += c
    counts = np.clip(counts, 1, None)
    weights = (counts.sum() / (num_classes * counts)) ** power
    if max_weight is not None:
        weights = np.clip(weights, None, max_weight)
    return torch.tensor(weights, dtype=torch.float32)
