"""
Shared ground-height grid utilities.

/api/process builds one of these from the classified LAS it already produces
during upload (no extra classification pass - it just reads the Classification==2
points that are already there), and saves it alongside the converted point cloud.
/api/drape-shp later reuses that saved grid to drape an uploaded SHP overlay onto
real ground height, without needing the original LAS again.
"""

import json
import subprocess

import numpy as np
import pdal

CELL_SIZE = 0.5  # meters
MAX_POINTS = 8_000_000  # hard cap on points read into memory, regardless of file size -
                         # without this, a large (150M+ point) file's full point array plus
                         # the groupby below can use 20GB+ RAM and get the whole server OOM-killed


def _total_point_count(las_path: str) -> int:
    # header-only read (fast even on multi-GB files, unlike materializing all points)
    result = subprocess.run(
        ["pdal", "info", "--metadata", las_path],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)["metadata"].get("count", 0)


def _grid_from_ground_points(arr):
    """arr: a structured array of already Classification==2 (ground-only) points.
    Returns (gx, gy, gz) per-cell median arrays, or None if arr is empty."""
    if len(arr) == 0:
        return None

    x, y, z = arr["X"], arr["Y"], arr["Z"]
    col = np.floor(x / CELL_SIZE).astype(np.int64)
    row = np.floor(y / CELL_SIZE).astype(np.int64)
    key = col * 10_000_000 + row

    order = np.argsort(key)
    key_sorted = key[order]
    x_s, y_s, z_s = x[order], y[order], z[order]
    _, start_idx = np.unique(key_sorted, return_index=True)
    groups = np.split(np.arange(len(key_sorted)), start_idx[1:])

    gx = np.empty(len(groups))
    gy = np.empty(len(groups))
    gz = np.empty(len(groups))
    for i, g in enumerate(groups):
        # median, not min/lowest-point: Classification==2 already excludes non-ground
        # objects (SMRF), so this just smooths residual per-point noise
        gx[i] = np.median(x_s[g])
        gy[i] = np.median(y_s[g])
        gz[i] = np.median(z_s[g])

    return gx, gy, gz


def build_ground_proxy(las_path: str):
    """las_path must already be ground-classified (Classification==2 present).
    Returns (gx, gy, gz) arrays, or None if there are no ground points."""
    total_points = _total_point_count(las_path)
    stride = max(1, total_points // MAX_POINTS)

    pipeline_stages = [{"type": "readers.las", "filename": las_path}]
    if stride > 1:
        pipeline_stages.append({"type": "filters.decimation", "step": stride})
    pipeline_stages.append({"type": "filters.range", "limits": "Classification[2:2]"})

    pipeline = pdal.Pipeline(json.dumps({"pipeline": pipeline_stages}))
    pipeline.execute()
    return _grid_from_ground_points(pipeline.arrays[0])


def save_ground_proxy(path, gx, gy, gz):
    np.savez(path, gx=gx, gy=gy, gz=gz)


def load_ground_proxy(path):
    data = np.load(path)
    return data["gx"], data["gy"], data["gz"]
