#!/usr/bin/env python3
"""
Converts point-wise Cylinder3D predictions (see run_inference.py's output
GeoJSON of Point features) into vector geometry - LineString for lane,
Polygon for crosswalk - matching the shape of the original HD map ground
truth, instead of a scattered point splatter.

Lane: rasterize predicted lane points to a binary grid, skeletonize (thin to a
1px-wide centerline), split into connected components, order each component's
pixels along its principal axis (lane stripes are locally near-straight, so
this is a simple stand-in for full skeleton-graph tracing), simplify.

Crosswalk: rasterize predicted crosswalk points to a binary grid, label
connected components, extract each component's polygon boundary directly from
the raster (rasterio.features.shapes), simplify.

Output is written as an actual ESRI Shapefile (via ogr2ogr, same as
drape_shp.py's output) so it can be loaded exactly like an HD map overlay -
through the existing sidebar "Import SHP overlay" flow or
examples/test_shp_overlay.html - no new frontend code needed.

Usage:
    python3 training/vectorize_predictions.py <predictions.geojson> <out.shp> --epsg EPSG:32652
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import rasterio.features
from affine import Affine
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, mapping, shape as shapely_shape
from skimage.morphology import disk, skeletonize

CELL_SIZE = 0.15
# 5-pixel (0.75m) skeleton fragments give a very noisy PCA direction estimate
# (measured: only ~42% of true dash-to-dash continuations passed a 12deg
# check because the short segments' own direction was unreliable, not
# because the gap itself was misaligned) - raising the floor to 10 cells
# (1.5m) trades some very short dashes for segments whose direction can
# actually be trusted for stitching.
MIN_LANE_CELLS = 10
MIN_CROSSWALK_CELLS = 20
LANE_SIMPLIFY_TOL = 0.3
CROSSWALK_SIMPLIFY_TOL = 0.2
DASH_MAX_GAP_M = 8.0  # bridge dash-to-dash gaps up to this far apart
DASH_MAX_ANGLE_DEG = 20.0  # ...as long as the two dashes point the same way
DASH_MAX_TOTAL_BEND_DEG = 25.0  # cap on cumulative drift along a stitched chain
STRAIGHT_FIT_RESIDUAL_TOL = 0.4  # perpendicular scatter (m) below which a chain collapses to a straight 2-point line


def load_points(geojson_path):
    d = json.loads(Path(geojson_path).read_text())
    lane, crosswalk = [], []
    for f in d["features"]:
        c = f["geometry"]["coordinates"]
        (lane if f["properties"]["class"] == "lane" else crosswalk).append(c)
    return np.array(lane), np.array(crosswalk)


def rasterize_points(points, cell_size, pad=3):
    xmin = points[:, 0].min() - pad * cell_size
    ymin = points[:, 1].min() - pad * cell_size
    xmax = points[:, 0].max() + pad * cell_size
    ymax = points[:, 1].max() + pad * cell_size
    width = max(1, int(np.ceil((xmax - xmin) / cell_size)))
    height = max(1, int(np.ceil((ymax - ymin) / cell_size)))
    transform = Affine(cell_size, 0, xmin, 0, -cell_size, ymax)

    grid = np.zeros((height, width), dtype=bool)
    inv = ~transform
    cols, rows = inv * (points[:, 0], points[:, 1])
    cols = np.clip(cols.astype(np.int64), 0, width - 1)
    rows = np.clip(rows.astype(np.int64), 0, height - 1)
    grid[rows, cols] = True
    return grid, transform


def pixel_to_world(rows, cols, transform):
    xs, ys = transform * (cols, rows)
    return xs, ys


def build_z_grid(points, shape, transform):
    """Mean Z per grid cell where points fall; cells with no original point in
    them (e.g. pixels only touched by closing/skeletonizing) get the nearest
    occupied cell's Z via a distance-transform nearest-fill. Without this, the
    extracted line/polygon geometry has no elevation and the SHP loader falls
    back to one flat height for the whole layer - usually well below the real
    road surface, so it renders hidden behind/under the point cloud."""
    height, width = shape
    inv = ~transform
    cols, rows = inv * (points[:, 0], points[:, 1])
    cols = np.clip(cols.astype(np.int64), 0, width - 1)
    rows = np.clip(rows.astype(np.int64), 0, height - 1)

    sum_z = np.zeros(shape)
    count = np.zeros(shape)
    np.add.at(sum_z, (rows, cols), points[:, 2])
    np.add.at(count, (rows, cols), 1)
    has_data = count > 0
    z_grid = np.zeros(shape)
    z_grid[has_data] = sum_z[has_data] / count[has_data]

    _, (nr, nc) = ndimage.distance_transform_edt(~has_data, return_indices=True)
    return z_grid[nr, nc]


def _extract_skeleton_segments(points, cell_size):
    """Per skeleton connected component, order pixels along the component's
    principal axis (stand-in for full skeleton-graph tracing - reasonable for
    the short, near-straight stripes skeletonize() produces here), smooth the
    ordered sequence to remove pixel-level zigzag, and return each as a
    lightweight segment dict with its endpoints/direction so segments can be
    stitched across dash gaps afterwards."""
    grid, transform = rasterize_points(points, cell_size)
    z_grid = build_z_grid(points, grid.shape, transform)
    # raw predicted-point rasterization is sparse/noisy (e.g. 21621 skeleton
    # fragments, mean size 3.6px, on one real test run) - a closing pass first
    # bridges small gaps so skeletonize produces contiguous stripes instead of
    # thousands of near-isolated noise specks (down to ~4000 components,
    # mean size 11.9px, after closing with a radius-2 disk)
    closed = ndimage.binary_closing(grid, structure=disk(2))
    skeleton = skeletonize(closed)

    labeled, n = ndimage.label(skeleton, structure=np.ones((3, 3)))
    # one bounding-box slice per label instead of a full-array np.where scan
    # per component - the latter is O(grid size) per component and was the
    # actual bottleneck (thousands of components x ~19M-cell array scans)
    objects = ndimage.find_objects(labeled)

    segments = []
    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        sub = labeled[slc] == comp_id
        if sub.sum() < MIN_LANE_CELLS:
            continue
        rows_local, cols_local = np.where(sub)
        rows = rows_local + slc[0].start
        cols = cols_local + slc[1].start

        pts = np.column_stack([cols, rows]).astype(float)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        proj = centered @ axis
        order = np.argsort(proj)

        rows_ord = rows[order].astype(float)
        cols_ord = cols[order].astype(float)
        zs = z_grid[rows[order], cols[order]]

        # raw skeleton pixels zigzag pixel-by-pixel along the stripe (classic
        # thinning artifact - looked like a "worm" instead of a clean line),
        # since skeletonize() traces every boundary wobble of the underlying
        # noisy point splatter. Smooth the ordered row/col sequence with a
        # Savitzky-Golay filter before projecting to world coords - this
        # removes the high-frequency jitter while keeping real curvature.
        if len(rows_ord) >= 7:
            window = min(11, len(rows_ord) - (1 - len(rows_ord) % 2))
            if window % 2 == 0:
                window -= 1
            if window >= 5:
                rows_ord = savgol_filter(rows_ord, window, 2, mode="nearest")
                cols_ord = savgol_filter(cols_ord, window, 2, mode="nearest")

        xs, ys = pixel_to_world(rows_ord, cols_ord, transform)
        coords = np.column_stack([xs, ys, zs])
        if len(coords) < 2:
            continue

        start, end = coords[0, :2], coords[-1, :2]
        direction = end - start
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            continue
        segments.append({"coords": coords, "start": start, "end": end, "direction": direction / norm})

    return segments


def _stitch_segments(segments, max_gap, max_angle_deg):
    """Real lane paint is dashed - each dash became its own skeleton
    component/segment above, so a single painted line still shows up as many
    disconnected short pieces (the earlier 'worm'/fragmentation complaint).
    Bridge segments across the dash gaps: build a graph linking segment
    endpoints ('ports') that are close together and whose outward direction
    is roughly opposite (i.e. the two segments continue in the same overall
    direction), then walk the resulting chains of at-most-degree-2 segments
    into ordered lists. Returns a list of chains, each a list of
    (segment_index, reversed) pairs in traversal order."""
    n = len(segments)
    if n == 0:
        return []

    endpoints = []
    for i, seg in enumerate(segments):
        endpoints.append((i, 0, seg["start"], -seg["direction"]))
        endpoints.append((i, 1, seg["end"], seg["direction"]))
    endpoint_pts = np.array([e[2] for e in endpoints])

    tree = cKDTree(endpoint_pts)
    pairs = tree.query_pairs(r=max_gap)

    max_angle_cos = np.cos(np.radians(max_angle_deg))
    candidates = []
    for a, b in pairs:
        i, ei, pi, di = endpoints[a]
        j, ej, pj, dj = endpoints[b]
        if i == j:
            continue
        # outward directions should be roughly opposite - a and b are the two
        # ends of a gap the vehicle drove straight/gently curved through
        cos_angle = float(np.dot(di, dj))
        if cos_angle > -max_angle_cos:
            continue
        dist = float(np.linalg.norm(pi - pj))
        candidates.append((dist, i, ei, j, ej))
    candidates.sort(key=lambda c: c[0])

    port_link = {}
    for dist, i, ei, j, ej in candidates:
        if (i, ei) in port_link or (j, ej) in port_link:
            continue
        port_link[(i, ei)] = (j, ej)
        port_link[(j, ej)] = (i, ei)

    visited = [False] * n
    chains = []
    for i in range(n):
        if visited[i]:
            continue
        chain = [(i, False)]
        visited[i] = True

        cur, cur_port = i, 1
        while (cur, cur_port) in port_link:
            nxt, nxt_port = port_link[(cur, cur_port)]
            if visited[nxt]:
                break
            chain.append((nxt, nxt_port == 1))
            visited[nxt] = True
            cur, cur_port = nxt, 1 - nxt_port

        cur, cur_port = i, 0
        while (cur, cur_port) in port_link:
            nxt, nxt_port = port_link[(cur, cur_port)]
            if visited[nxt]:
                break
            chain.insert(0, (nxt, nxt_port == 0))
            visited[nxt] = True
            cur, cur_port = nxt, 1 - nxt_port

        chains.append(chain)

    return chains


def _split_bendy_chains(segments, chains, max_total_bend_deg):
    """Each stitch link only checks the angle between the two dashes it joins
    (<=DASH_MAX_ANGLE_DEG), but small per-junction misalignments can all lean
    the same way and accumulate along a long chain - e.g. ten links each 10
    degrees off, drifting the same direction, silently turn a chain 100
    degrees overall (this is what produced real outliers like an 8m bulge
    over a 16m chord: not noise, but an accumulated bend, often from
    bridging onto a different road's line near an intersection). Re-split
    any chain once its direction has drifted more than max_total_bend_deg
    from where that (sub-)chain started."""
    result = []
    for chain in chains:
        if len(chain) <= 1:
            result.append(chain)
            continue
        cur = [chain[0]]
        idx0, rev0 = chain[0]
        base_dir = -segments[idx0]["direction"] if rev0 else segments[idx0]["direction"]
        for idx, rev in chain[1:]:
            d = -segments[idx]["direction"] if rev else segments[idx]["direction"]
            cos_a = max(-1.0, min(1.0, float(np.dot(base_dir, d))))
            angle = np.degrees(np.arccos(cos_a))
            if angle > max_total_bend_deg:
                result.append(cur)
                cur = [(idx, rev)]
                base_dir = d
            else:
                cur.append((idx, rev))
        result.append(cur)
    return result


def _fit_chain(segments, chain):
    coords = np.vstack([
        segments[idx]["coords"][::-1] if rev else segments[idx]["coords"]
        for idx, rev in chain
    ])
    if len(coords) < 2:
        return None

    xy = coords[:, :2]
    z = coords[:, 2]
    centroid = xy.mean(axis=0)
    centered = xy - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    proj = centered @ axis
    order = np.argsort(proj)

    if vt.shape[0] > 1:
        residual = centered @ vt[1]
        residual_rms = float(np.sqrt(np.mean(residual ** 2)))
    else:
        residual_rms = 0.0

    if residual_rms <= STRAIGHT_FIT_RESIDUAL_TOL:
        # near-straight overall (single dash, or a stitched chain of dashes
        # that line up on one road stripe) - collapse to its two extreme
        # points instead of following every intermediate pixel, which is what
        # actually removes the "worm" look rather than just smoothing it
        i0, i1 = order[0], order[-1]
        return LineString([tuple(xy[i0]) + (float(z[i0]),), tuple(xy[i1]) + (float(z[i1]),)])

    # chain has real curvature (e.g. follows a curved road over a long
    # stitched run) - keep it as a simplified polyline instead of forcing a
    # straight line that would cut the corner. Each dash was already smoothed
    # individually, but stitching many dashes end-to-end re-exposes each
    # dash's own residual noise as a periodic zigzag at the dash-spacing
    # wavelength (visible as a regular sawtooth once dozens of dashes are
    # chained together) - a wide-window smoothing pass over the *whole*
    # chain (not just per-dash) removes that, while a real road curve's much
    # longer radius survives fine.
    xy_ord, z_ord = xy[order].copy(), z[order]
    if len(xy_ord) >= 15:
        window = min(31, len(xy_ord) - (1 - len(xy_ord) % 2))
        if window % 2 == 0:
            window -= 1
        if window >= 7:
            xy_ord[:, 0] = savgol_filter(xy_ord[:, 0], window, 2, mode="nearest")
            xy_ord[:, 1] = savgol_filter(xy_ord[:, 1], window, 2, mode="nearest")
    line_2d = LineString(list(zip(xy_ord[:, 0].tolist(), xy_ord[:, 1].tolist()))).simplify(LANE_SIMPLIFY_TOL)
    if line_2d.is_empty or line_2d.length == 0:
        return None
    nn_tree = cKDTree(xy_ord)
    _, nn_idx = nn_tree.query(np.array(line_2d.coords))
    return LineString([(x, y, float(z_ord[i])) for (x, y), i in zip(line_2d.coords, nn_idx)])


def extract_lane_lines(points, cell_size=CELL_SIZE):
    segments = _extract_skeleton_segments(points, cell_size)
    chains = _stitch_segments(segments, max_gap=DASH_MAX_GAP_M, max_angle_deg=DASH_MAX_ANGLE_DEG)
    chains = _split_bendy_chains(segments, chains, max_total_bend_deg=DASH_MAX_TOTAL_BEND_DEG)

    lines = []
    for chain in chains:
        line = _fit_chain(segments, chain)
        if line is not None:
            lines.append(line)
    return lines


def _attach_z(poly, z_grid, transform):
    inv = ~transform
    height, width = z_grid.shape

    def ring_with_z(coords):
        out = []
        for x, y in coords:
            col, row = inv * (x, y)
            col = int(np.clip(col, 0, width - 1))
            row = int(np.clip(row, 0, height - 1))
            out.append((x, y, float(z_grid[row, col])))
        return out

    exterior = ring_with_z(poly.exterior.coords)
    interiors = [ring_with_z(r.coords) for r in poly.interiors]
    return Polygon(exterior, interiors)


def extract_crosswalk_polygons(points, cell_size=CELL_SIZE):
    grid, transform = rasterize_points(points, cell_size)
    z_grid = build_z_grid(points, grid.shape, transform)
    # close small gaps from sensor dropout before labeling
    closed = ndimage.binary_closing(grid, structure=np.ones((3, 3)))
    labeled, n = ndimage.label(closed, structure=np.ones((3, 3)))
    objects = ndimage.find_objects(labeled)

    polygons = []
    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        # transform offset by the slice's start so per-component pixel
        # coordinates (relative to slc) still map to the right world position
        sub_transform = transform * Affine.translation(slc[1].start, slc[0].start)
        mask = labeled[slc] == comp_id
        if mask.sum() < MIN_CROSSWALK_CELLS:
            continue
        for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=sub_transform):
            if value != 1:
                continue
            poly = shapely_shape(geom).simplify(CROSSWALK_SIMPLIFY_TOL)
            if not poly.is_empty and poly.area > 0:
                polygons.append(_attach_z(poly, z_grid, transform))

    return polygons


def _write_one_shapefile(features, out_path, epsg):
    geojson = {"type": "FeatureCollection", "features": features}
    with tempfile.TemporaryDirectory() as tmp:
        gj_path = Path(tmp) / "layer.geojson"
        gj_path.write_text(json.dumps(geojson))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ogr2ogr", "-f", "ESRI Shapefile", str(out_path), str(gj_path), "-a_srs", epsg],
            check=True, capture_output=True, text=True,
        )


def write_shapefiles(lines, polygons, out_path_stem, epsg):
    """Shapefile format requires one geometry type per file, so lines and
    polygons can't share a single .shp - writes '<stem>_lane.shp' and
    '<stem>_crosswalk.shp' as needed. Returns the list of paths written."""
    written = []
    if lines:
        path = f"{out_path_stem}_lane.shp"
        features = [{"type": "Feature", "geometry": mapping(l), "properties": {"class": "lane"}} for l in lines]
        _write_one_shapefile(features, path, epsg)
        written.append(path)
    if polygons:
        path = f"{out_path_stem}_crosswalk.shp"
        features = [{"type": "Feature", "geometry": mapping(p), "properties": {"class": "crosswalk"}} for p in polygons]
        _write_one_shapefile(features, path, epsg)
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions_geojson")
    ap.add_argument("out_shp_stem", help="output path without extension; writes '<stem>_lane.shp' / '<stem>_crosswalk.shp'")
    ap.add_argument("--epsg", default="EPSG:32652")
    args = ap.parse_args()

    lane_points, crosswalk_points = load_points(args.predictions_geojson)
    print(f"loaded {len(lane_points)} lane points, {len(crosswalk_points)} crosswalk points")

    lines = extract_lane_lines(lane_points) if len(lane_points) > 0 else []
    print(f"extracted {len(lines)} lane lines")

    polygons = extract_crosswalk_polygons(crosswalk_points) if len(crosswalk_points) > 0 else []
    print(f"extracted {len(polygons)} crosswalk polygons")

    if not lines and not polygons:
        raise SystemExit("no vector features extracted - nothing to write")

    written = write_shapefiles(lines, polygons, args.out_shp_stem, args.epsg)
    for p in written:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
