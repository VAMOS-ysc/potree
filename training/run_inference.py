#!/usr/bin/env python3
"""
Runs the trained Cylinder3D model over every tile belonging to one source_id
(as prepared by prepare_dataset.py / found in <data_root>/velodyne/<source_id>_*.bin),
reassembles predictions into real-world (UTM) coordinates using offsets.json, and
writes the predicted lane/crosswalk points (background points are dropped - we only
care about visualizing the two rare classes) to a GeoJSON FeatureCollection of
Point features with a "class" property ("lane"/"crosswalk"), in the same EPSG as
the source point cloud.

That GeoJSON is meant to be loaded the same way the HD map SHP overlays already
are (see examples/test_shp_overlay.html, src/loader/ShapefileLoader.js) - raw
world (UTM) coordinates placed directly as scene position, same as the existing
drape_shp.py output, so it lines up with the point cloud without any extra
transform.

Usage:
    python3 training/run_inference.py <source_id> <out_geojson> \
        --config configs/cylinder3d_lane3d.py \
        --checkpoint work_dirs/cylinder3d_lane3d_full/epoch_30.pth \
        --data-root data/lane3d
"""
import argparse
import json
from pathlib import Path

import numpy as np
from mmdet3d.apis import inference_segmentor, init_model

CLASS_NAMES = {1: "lane", 2: "crosswalk"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source_id")
    ap.add_argument("out_geojson")
    ap.add_argument("--config", default="configs/cylinder3d_lane3d.py")
    ap.add_argument("--checkpoint", default="work_dirs/cylinder3d_lane3d_full/epoch_30.pth")
    ap.add_argument("--data-root", default="data/lane3d")
    ap.add_argument("--epsg", default="EPSG:32652")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    offsets = json.loads((data_root / "offsets.json").read_text())

    tiles = sorted((data_root / "velodyne").glob(f"{args.source_id}_*.bin"))
    if not tiles:
        raise SystemExit(f"no tiles found for source_id={args.source_id!r} under {data_root}/velodyne")
    print(f"running inference on {len(tiles)} tiles for {args.source_id} ...")

    model = init_model(args.config, args.checkpoint, device="cuda:0")

    features = []
    class_counts = {1: 0, 2: 0}
    for tile_path in tiles:
        result, _ = inference_segmentor(model, str(tile_path))
        pred = result.pred_pts_seg["pts_semantic_mask"]
        pred = pred.cpu().numpy() if hasattr(pred, "cpu") else np.asarray(pred)

        keep = pred != 0
        if not keep.any():
            continue

        pts = np.fromfile(tile_path, dtype=np.float32).reshape(-1, 4)
        x0, y0, z0 = offsets[tile_path.stem]

        wx = pts[keep, 0] + x0
        wy = pts[keep, 1] + y0
        wz = pts[keep, 2] + z0
        cls = pred[keep]

        for x, y, z, c in zip(wx, wy, wz, cls):
            c = int(c)
            class_counts[c] += 1
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y), float(z)]},
                "properties": {"class": CLASS_NAMES[c]},
            })

    geojson = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": args.epsg}},
        "features": features,
    }
    Path(args.out_geojson).write_text(json.dumps(geojson))
    print(f"wrote {len(features)} predicted points to {args.out_geojson}")
    print(f"  lane: {class_counts[1]}, crosswalk: {class_counts[2]}")


if __name__ == "__main__":
    main()
