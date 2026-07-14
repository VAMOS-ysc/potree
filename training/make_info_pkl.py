#!/usr/bin/env python3
"""
Builds the train/val info .pkl files MMDetection3D's SemanticKittiDataset expects,
by scanning a flat <data_root>/velodyne/*.bin + <data_root>/labels/*.label
directory (as written by prepare_dataset.py) - unlike real SemanticKITTI, which
uses a sequences/00../21 layout with a fixed train/val split by sequence, our
tiles all live in one flat directory tagged '<source_id>_<tile_id>.bin', so the
split is done by source_id (whole source LAS files held out for val) to avoid
leaking near-identical/overlapping tiles between train and val.

Usage:
    python3 training/make_info_pkl.py <data_root> --val-source-ids 19-002,18-003
"""
import argparse
import re
from pathlib import Path

import mmengine

TILE_RE = re.compile(r"^(.*)_(\d{12})$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_root")
    ap.add_argument("--val-source-ids", default="",
                     help="comma-separated source_id prefixes (e.g. LAS filename stems) held out for val")
    ap.add_argument("--pkl-prefix", default="lane3d")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    val_ids = set(s for s in args.val_source_ids.split(",") if s)

    bin_files = sorted((data_root / "velodyne").glob("*.bin"))
    if not bin_files:
        raise SystemExit(f"no .bin files found under {data_root / 'velodyne'}")

    by_source = {}
    for f in bin_files:
        m = TILE_RE.match(f.stem)
        if not m:
            raise SystemExit(f"unexpected tile filename (expected '<source_id>_<12 digits>'): {f.name}")
        source_id = m.group(1)
        by_source.setdefault(source_id, []).append(f.stem)

    print(f"source files found: {sorted(by_source.keys())}")
    unknown_val = val_ids - set(by_source.keys())
    if unknown_val:
        raise SystemExit(f"--val-source-ids has unknown source_id(s): {unknown_val}")

    splits = {"train": [], "val": []}
    for source_id, tiles in by_source.items():
        split = "val" if source_id in val_ids else "train"
        splits[split].extend(tiles)

    for split, tiles in splits.items():
        data_list = [{
            "lidar_points": {"lidar_path": f"velodyne/{t}.bin", "num_pts_feats": 4},
            "pts_semantic_mask_path": f"labels/{t}.label",
            "sample_id": t,
        } for t in tiles]
        info = {"metainfo": {"DATASET": "Lane3D"}, "data_list": data_list}
        out_path = data_root / f"{args.pkl_prefix}_infos_{split}.pkl"
        mmengine.dump(info, out_path)
        print(f"{split}: {len(tiles)} tiles -> {out_path}")


if __name__ == "__main__":
    main()
