#!/usr/bin/env python3
"""
Drape a shapefile's vertices onto a point cloud's real ground height (via a
precomputed ground-height grid - see ground_proxy.py), instead of placing the
whole overlay at one flat Z (which looks wrong wherever the road isn't flat).

Also usable as a CLI for one-off testing:
    python3 webapp/drape_shp.py <input.shp> <ground_proxy.npz> <target_epsg> <output.shp> \
        <xmin> <ymin> <xmax> <ymax>

<xmin> <ymin> <xmax> <ymax> is the point cloud's bounding box (target_epsg units,
e.g. from metadata.json's boundingBox) - the HD map/shapefile usually covers a much
wider area than any one point cloud, and draping features far outside the point
cloud's extent just finds whatever ground sample happens to be nearest, which is
meaningless - crop to the point cloud's extent (+ a small buffer) first.

Reprojection (source SHP CRS -> target_epsg) and the final GeoJSON->SHP write both
go through `ogr2ogr`, same as the existing /api/export-shp endpoint - no new
geospatial dependency beyond what's already used elsewhere here.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.spatial import cKDTree

from ground_proxy import load_ground_proxy

CROP_BUFFER = 20.0  # meters
MAX_GROUND_DISTANCE = 50.0  # meters - beyond this, there's no real nearby ground sample
SMOOTH_WINDOW = 5  # vertices - kills single-point spikes (fallback-mean transitions etc.)


def drape_line(coords, tree, gz, skip_counter):
    """coords: list of [x,y] (or [x,y,z]) vertices of ONE line/ring. Returns the same
    vertices with z looked up per-point, then median-smoothed along the sequence so a
    single bad sample doesn't show up as a sharp spike - real road elevation changes
    gradually."""
    xy = np.array([[c[0], c[1]] for c in coords])
    d, i = tree.query(xy)
    z = gz[i].copy()

    far = d > MAX_GROUND_DISTANCE
    if far.any():
        skip_counter[0] += int(far.sum())
        z[far] = gz.mean()

    if len(z) >= 3:
        z = median_filter(z, size=min(SMOOTH_WINDOW, len(z)), mode="nearest")

    return [[c[0], c[1], float(z[j])] for j, c in enumerate(coords)]


def drape_feature_geometry(geometry, tree, gz, skip_counter):
    """Dispatches on the GeoJSON geometry type rather than guessing from array
    nesting depth - Point's `coordinates` is a single [x,y] tuple, not a list of
    them, which broke draping for point layers (e.g. traffic lights) when this
    used to infer structure from `coordinates`."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]

    if gtype in ("Point",):
        geometry["coordinates"] = drape_line([coords], tree, gz, skip_counter)[0]
    elif gtype in ("LineString", "MultiPoint"):
        geometry["coordinates"] = drape_line(coords, tree, gz, skip_counter)
    elif gtype in ("Polygon", "MultiLineString"):
        geometry["coordinates"] = [drape_line(ring, tree, gz, skip_counter) for ring in coords]
    elif gtype == "MultiPolygon":
        geometry["coordinates"] = [
            [drape_line(ring, tree, gz, skip_counter) for ring in poly] for poly in coords
        ]
    else:
        raise ValueError(f"unsupported geometry type: {gtype}")


def drape_shapefile(shp_path: str, ground_proxy_path: str, target_epsg: str,
                     bbox: tuple, out_path: str, source_srs: str | None = None) -> dict:
    """bbox = (xmin, ymin, xmax, ymax) in target_epsg units. Writes the draped
    shapefile to out_path. Returns a small stats dict for logging/reporting.

    source_srs: pass this when the shapefile has no .prj (ogr2ogr can't reproject
    without knowing the source CRS) - e.g. "EPSG:4326" if the HD map's other layers
    are all known to be WGS84 lon/lat."""
    xmin, ymin, xmax, ymax = bbox
    clip_bbox = [xmin - CROP_BUFFER, ymin - CROP_BUFFER, xmax + CROP_BUFFER, ymax + CROP_BUFFER]

    with tempfile.TemporaryDirectory() as tmp:
        geojson_path = Path(tmp) / "reproj.geojson"
        cmd = ["ogr2ogr", "--config", "SHAPE_RESTORE_SHX", "YES",
               "-f", "GeoJSON", str(geojson_path), shp_path]
        if source_srs:
            cmd += ["-s_srs", source_srs]
        cmd += ["-t_srs", target_epsg, "-clipdst", *(str(v) for v in clip_bbox)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        geojson = json.loads(geojson_path.read_text())

    if len(geojson["features"]) == 0:
        return {"features_in": 0, "features_out": 0, "vertices_no_ground": 0}

    gx, gy, gz = load_ground_proxy(ground_proxy_path)
    tree = cKDTree(np.column_stack([gx, gy]))

    skipped = [0]
    for feature in geojson["features"]:
        drape_feature_geometry(feature["geometry"], tree, gz, skipped)

    with tempfile.TemporaryDirectory() as tmp:
        draped_geojson = Path(tmp) / "draped.geojson"
        draped_geojson.write_text(json.dumps(geojson))

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ogr2ogr", "-f", "ESRI Shapefile", "-dim", "XYZ", out_path, str(draped_geojson), "-a_srs", target_epsg],
            check=True, capture_output=True, text=True,
        )

    return {
        "features_in": len(geojson["features"]),
        "features_out": len(geojson["features"]),
        "vertices_no_ground": skipped[0],
    }


def main():
    if len(sys.argv) != 9:
        print(__doc__)
        sys.exit(1)

    shp_path, ground_proxy_path, target_epsg, out_path = sys.argv[1:5]
    bbox = tuple(float(v) for v in sys.argv[5:9])

    stats = drape_shapefile(shp_path, ground_proxy_path, target_epsg, bbox, out_path)
    print(f"wrote {out_path}: {stats}", file=sys.stderr)


if __name__ == "__main__":
    main()
