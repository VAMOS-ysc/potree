#!/usr/bin/env python3
"""
Prepares SemanticKITTI-format training tiles (velodyne/*.bin, labels/*.label) for
Cylinder3D, using ground-classified LiDAR points from a raw LAS file and HD map
SHP layers (B2 lane lines, B3 crosswalk polygons) as ground truth via buffer-based
point labeling.

Class scheme: 0=background(other ground), 1=lane, 2=crosswalk

HD map Kind-code semantics are not documented anywhere in the source data - they
were reverse-engineered by cross-referencing SHP feature IDs against an OpenDRIVE
export (AYG_DNA_PCN.xml) that carries explicit type="crosswalk"/"stopline" tags for
a subset of features:
  - B2_SURFACELINEMARK Kind=530 -> stopline (578/578 confirmed cases) - excluded from
    the lane class, since a stop line isn't a lane marking
  - B3_SURFACEMARK Kind=5321 -> crosswalk (475/475 confirmed cases) - the only Kind
    used for the crosswalk class; other B3 Kind values have no confirmed meaning and
    are left unlabeled (background) rather than guessed

Usage:
    python3 training/prepare_dataset.py <raw_las> <hdmap_dir> <out_dir> [--tile-size 30]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pdal
import rasterio.features
from affine import Affine
from shapely.geometry import shape
from shapely.ops import unary_union

CLASS_BACKGROUND = 0
CLASS_LANE = 1
CLASS_CROSSWALK = 2

LANE_BUFFER_M = 0.15   # half-width around the digitized lane centerline
RASTER_CELL_M = 0.05   # label raster resolution
CROP_BUFFER_M = 20.0   # HD map crop margin around the point cloud's bbox

# SMRF's peak memory scales with the RAW (pre-classification) point count, and
# measured at ~17.2GB RSS for a 194M-point file in isolation - on a 31GB machine,
# also running everything else the user has open, that's too close to the edge.
# 07-004.las (236.7M points, the largest of the 14 source files) OOM-killed
# (SIGKILL) partway through a batch run and the box needed a hard reboot -
# without this, retrying the batch would just repeat the crash. Cap the raw
# point count fed into SMRF via decimation, same pattern as ground_proxy.py's
# MAX_POINTS fix from an earlier OOM incident this session.
MAX_RAW_POINTS_FOR_SMRF = 120_000_000

TILE_SIZE_M = 30.0
TILE_MIN_POINTS = 3000
# a real SemanticKITTI scan is ~120k points; our tiles are built from a
# multi-pass aggregated corridor map and can run 500k-1M+ points, ~5-8x denser -
# subsample down to something closer to the scale Cylinder3D/the voxel grid is
# tuned for, and to keep per-tile training speed/memory reasonable.
# 150k OOM'd an 11GB GPU even at batch_size=1 (smoke test); 50k leaves headroom
# for a real batch size.
MAX_POINTS_PER_TILE = 50_000


def get_epsg(las_path: str) -> str:
    result = subprocess.run(["pdal", "info", "--metadata", las_path], capture_output=True, text=True, check=True)
    meta = json.loads(result.stdout)["metadata"]
    srs = meta.get("srs", {})
    for c in srs.get("json", {}).get("components", [srs.get("json", {})]):
        code = c.get("id", {}).get("code")
        if code:
            return f"EPSG:{code}"
    raise RuntimeError(f"could not determine EPSG for {las_path}")


def _total_point_count(las_path: str) -> int:
    # header-only read (fast even on multi-GB files, unlike materializing all points)
    result = subprocess.run(["pdal", "info", "--metadata", las_path], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["metadata"].get("count", 0)


def classify_ground_points(las_path: str):
    """Runs SMRF and returns ground-only (Classification==2) points as a
    structured array with X, Y, Z, Intensity fields. Decimates the raw input
    first if it's large enough that SMRF's peak memory would risk an OOM (see
    MAX_RAW_POINTS_FOR_SMRF)."""
    total_points = _total_point_count(las_path)
    stride = max(1, total_points // MAX_RAW_POINTS_FOR_SMRF)

    pipeline_stages = [{"type": "readers.las", "filename": las_path}]
    if stride > 1:
        print(f"  {total_points} raw points > cap - decimating by stride {stride} before SMRF",
              file=sys.stderr)
        pipeline_stages.append({"type": "filters.decimation", "step": stride})
    pipeline_stages.append({"type": "filters.smrf"})
    pipeline_stages.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline = pdal.Pipeline(json.dumps({"pipeline": pipeline_stages}))
    pipeline.execute()
    return pipeline.arrays[0]


def load_hdmap_shapes(hdmap_dir: str, target_epsg: str, bbox):
    """Returns (lane_geoms, crosswalk_geoms): lists of shapely geometries in
    target_epsg, cropped to bbox + CROP_BUFFER_M, filtered by the Kind-code rules
    documented in the module docstring. lane_geoms are already buffered
    (LANE_BUFFER_M); crosswalk_geoms are the raw polygons (already have area)."""
    xmin, ymin, xmax, ymax = bbox
    clip = [xmin - CROP_BUFFER_M, ymin - CROP_BUFFER_M, xmax + CROP_BUFFER_M, ymax + CROP_BUFFER_M]

    def reproject_crop(shp_path):
        # some HD map layers (B3 in practice) ship without a .prj - ogr2ogr can't
        # reproject without a source CRS, so assume WGS84 lon/lat when it's missing
        # (matches every other layer in this export and webapp/drape_shp.py's fallback)
        source_srs = None if Path(shp_path).with_suffix(".prj").exists() else "EPSG:4326"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.geojson"
            cmd = ["ogr2ogr", "--config", "SHAPE_RESTORE_SHX", "YES", "-f", "GeoJSON", str(out), shp_path]
            if source_srs:
                cmd += ["-s_srs", source_srs]
            cmd += ["-t_srs", target_epsg, "-clipdst", *(str(v) for v in clip)]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return json.loads(out.read_text())

    b2 = reproject_crop(str(Path(hdmap_dir) / "B2_SURFACELINEMARK.shp"))
    b3 = reproject_crop(str(Path(hdmap_dir) / "B3_SURFACEMARK.shp"))

    lane_geoms = []
    for f in b2["features"]:
        if f["properties"].get("Kind") == "530":  # stopline, not a lane marking
            continue
        geom = shape(f["geometry"])
        if not geom.is_empty:
            lane_geoms.append(geom.buffer(LANE_BUFFER_M))

    crosswalk_geoms = []
    for f in b3["features"]:
        if f["properties"].get("Kind") != "5321":  # only the confirmed crosswalk code
            continue
        geom = shape(f["geometry"])
        if not geom.is_empty:
            crosswalk_geoms.append(geom)

    return lane_geoms, crosswalk_geoms


def rasterize_labels(lane_geoms, crosswalk_geoms, bbox, cell_size):
    xmin, ymin, xmax, ymax = bbox
    width = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    height = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    transform = Affine(cell_size, 0, xmin, 0, -cell_size, ymax)

    label_grid = np.zeros((height, width), dtype=np.uint8)
    if lane_geoms:
        lane_union = unary_union(lane_geoms)
        rasterio.features.rasterize([(lane_union, CLASS_LANE)], out=label_grid, transform=transform)
    if crosswalk_geoms:
        # rasterized after lanes so crosswalk wins on any overlap (e.g. a lane
        # line's buffer clipping into a crosswalk's edge)
        cw_union = unary_union(crosswalk_geoms)
        rasterio.features.rasterize([(cw_union, CLASS_CROSSWALK)], out=label_grid, transform=transform)

    return label_grid, transform


def sample_labels(points_x, points_y, label_grid, transform):
    inv = ~transform
    cols, rows = inv * (points_x, points_y)
    cols = np.clip(cols.astype(np.int64), 0, label_grid.shape[1] - 1)
    rows = np.clip(rows.astype(np.int64), 0, label_grid.shape[0] - 1)
    return label_grid[rows, cols]


def tile_and_write(arr, labels, out_dir, tile_size, min_points, source_id):
    """Writes one .bin/.label pair per tile, named '<source_id>_<tile_id>'.

    Points are written relative to each tile's own centroid (x0, y0) and a z0
    reference (median ground height in the tile), not raw UTM coordinates - UTM
    easting/northing are on the order of 1e5-1e7, wildly outside the small-scale
    (tens of meters) point_cloud_range Cylinder3D/SemanticKITTI configs expect, and
    would put every point far outside any reasonable voxel grid. The per-tile
    offset is recorded in offsets.json (appended) so tile predictions can be
    reprojected back to world coordinates later if needed."""
    out_dir = Path(out_dir)
    (out_dir / "velodyne").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    x, y, z, intensity = arr["X"], arr["Y"], arr["Z"], arr["Intensity"]
    col = np.floor((x - x.min()) / tile_size).astype(np.int64)
    row = np.floor((y - y.min()) / tile_size).astype(np.int64)
    tile_id = col * 100_000 + row

    offsets_path = out_dir / "offsets.json"
    offsets = json.loads(offsets_path.read_text()) if offsets_path.exists() else {}

    tiles_written = 0
    class_counts = {0: 0, 1: 0, 2: 0}
    for tid in np.unique(tile_id):
        mask = tile_id == tid
        n = int(mask.sum())
        if n < min_points:
            continue

        tx, ty = x[mask], y[mask]
        tz = z[mask]
        x0, y0, z0 = float(tx.mean()), float(ty.mean()), float(np.median(tz))

        pts = np.stack([tx - x0, ty - y0, tz - z0, intensity[mask].astype(np.float32)],
                        axis=1).astype(np.float32)
        pts[:, 3] = pts[:, 3] / 255.0  # this data's raw Intensity is 8-bit (checked empirically, not 16-bit)
        lbl = labels[mask].astype(np.uint32)

        if len(pts) > MAX_POINTS_PER_TILE:
            keep = np.random.choice(len(pts), MAX_POINTS_PER_TILE, replace=False)
            pts, lbl = pts[keep], lbl[keep]

        name = f"{source_id}_{tid:012d}"
        pts.tofile(out_dir / "velodyne" / f"{name}.bin")
        lbl.tofile(out_dir / "labels" / f"{name}.label")
        offsets[name] = [x0, y0, z0]
        tiles_written += 1
        for c in (0, 1, 2):
            class_counts[c] += int((lbl == c).sum())

    offsets_path.write_text(json.dumps(offsets))
    return tiles_written, class_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("las_path")
    ap.add_argument("hdmap_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--tile-size", type=float, default=TILE_SIZE_M)
    ap.add_argument("--source-id", default=None,
                     help="prefix for output tile filenames; defaults to the LAS filename stem")
    args = ap.parse_args()
    source_id = args.source_id or Path(args.las_path).stem

    print(f"[1/4] classifying ground points from {args.las_path} ...", file=sys.stderr)
    arr = classify_ground_points(args.las_path)
    print(f"  {len(arr)} ground points", file=sys.stderr)

    epsg = get_epsg(args.las_path)
    bbox = (float(arr["X"].min()), float(arr["Y"].min()), float(arr["X"].max()), float(arr["Y"].max()))
    print(f"[2/4] loading HD map shapes (epsg={epsg}, bbox={bbox}) ...", file=sys.stderr)
    lane_geoms, crosswalk_geoms = load_hdmap_shapes(args.hdmap_dir, epsg, bbox)
    print(f"  {len(lane_geoms)} lane features, {len(crosswalk_geoms)} crosswalk features", file=sys.stderr)

    print("[3/4] rasterizing labels and sampling per point ...", file=sys.stderr)
    label_grid, transform = rasterize_labels(lane_geoms, crosswalk_geoms, bbox, RASTER_CELL_M)
    labels = sample_labels(arr["X"], arr["Y"], label_grid, transform)

    print("[4/4] tiling and writing SemanticKITTI-format files ...", file=sys.stderr)
    n_tiles, class_counts = tile_and_write(arr, labels, args.out_dir, args.tile_size, TILE_MIN_POINTS, source_id)

    print(f"done: {n_tiles} tiles written to {args.out_dir}", file=sys.stderr)
    print(f"class point counts: background={class_counts[0]} lane={class_counts[1]} crosswalk={class_counts[2]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
