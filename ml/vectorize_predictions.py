#!/usr/bin/env python3
"""
예측된 클래스 래스터(ml/infer.py --out-mask 결과, mask.tif와 같은 uint8 class-id
포맷)를 QGIS에서 보는 정밀도로지도 SHP처럼 깔끔한 벡터(차선/정지선=LineString,
횡단보도=Polygon)로 변환한다. potree 점 스플래터(mask_to_geojson) 대신 쓰는,
"제대로 된" 결과물용 경로.

cylinder3d 브랜치의 training/vectorize_predictions.py(포인트 목록 입력, SHP 출력)을
포팅하되 두 가지를 바꿨다:
  1. 입력이 이미 래스터(예측 mask)라서, 포인트로 흩뿌렸다가 다시 격자화하는
     손실 있는 왕복을 안 하고 래스터를 직접 skeletonize/polygonize한다 -
     ml/infer.py가 클래스당 8만 점으로 서브샘플링한 뒤 0.15m 격자로 재격자화하면
     원본 0.05m 해상도 예측의 상당 부분을 그냥 버리게 됨.
  2. 출력을 SHP가 아니라 GeoJSON(LineString/Polygon, Z 포함)으로 써서
     examples/test_predictions.html에서 바로 렌더링 - ShapefileLoader.js가 이
     브랜치(UNET)에는 없어서 SHP 왕복이 필요 없게 함.

lane/stop 둘 다 "선" 클래스로 스켈레톤 추출한다 - B2_SURFACELINEMARK 자체가
LineString이라 (rasterize.py에서 폭을 넓혀 학습시켰을 뿐) 이게 맞는 표현이다.
정지선은 점선 이어붙이기(dash stitching)가 필요 없을 가능성이 높지만
(단일 굵은 바) 파라미터 하나 다르게 주는 것 말고 로직을 따로 만들 필요는 없어서
그대로 재사용한다.

사용 예:
    python ml/vectorize_predictions.py \\
        --scene-dir out/09-002 --mask pred_mask_v4.tif \\
        --out ../pointclouds/09-002_.../predictions_v4_vectors.geojson
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio.features
import tifffile
from affine import Affine
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon, mapping, shape as shapely_shape
from skimage.morphology import disk, skeletonize

# cylinder3d의 원본 상수는 0.15m 격자(포인트 재격자화 결과) 기준 픽셀 개수였다 -
# 여기선 실제 씬 해상도(보통 0.05m)에서 직접 돌기 때문에, 물리 단위(m)로 두고
# 실행 시점에 해상도로 나눠 픽셀 개수/윈도 크기로 변환한다.
MIN_LANE_LENGTH_M = 1.5
MIN_CROSSWALK_AREA_M2 = 0.45
LANE_SIMPLIFY_TOL = 0.3
CROSSWALK_SIMPLIFY_TOL = 0.2
DASH_MAX_GAP_M = 8.0
DASH_MAX_ANGLE_DEG = 20.0
DASH_MAX_TOTAL_BEND_DEG = 25.0
STRAIGHT_FIT_RESIDUAL_TOL = 0.4
SEGMENT_SMOOTH_WINDOW_M = 1.65   # per-dash smoothing (was window=11 @ 0.15m)
CHAIN_SMOOTH_WINDOW_M = 4.65     # whole-chain smoothing (was window=31 @ 0.15m)

CLASS_NAME_TO_GEOM = {1: ("lane", "line"), 2: ("crosswalk", "polygon")}


def _odd_window(length_m: float, resolution: float, min_len: int) -> int:
    w = round(length_m / resolution)
    if w % 2 == 0:
        w += 1
    return max(w, min_len)


GROUND_HEIGHT_BLOCK_M = 2.0  # block size for the robust ground-height estimate below


def _robust_ground_height(height: np.ndarray, resolution: float) -> np.ndarray:
    """rasterize.py's height.tif is the mean Z of ALL LAS returns in each cell
    (no ground classification - the raw source LAS's Classification field is
    all-zero, see rasterize.py), so a cell under a tree branch, overhead wire,
    or passing car gets pulled up by those returns. Confirmed on real data:
    at ground-truth lane_line pixels in one scene, height ranged 58.96-83.5m
    (a 24.5m spread) despite it being one fairly flat road - showed up as
    lines/polygons visibly spiking upward in potree.

    Contamination is one-sided (extra returns can only sit above the true
    road surface, never below it), so the minimum in a small neighborhood is
    a robust ground estimate - as long as some genuine ground return exists
    nearby, which holds for open road (tested: max dropped from 83.5m to
    ~75m, while the 58-75m range - real terrain variation - was untouched).
    A sliding-window min/percentile_filter over the full ~190M-pixel raster
    was too slow (still running after 5+ min for one window size) - block-
    reduce to a coarse grid (min per non-overlapping block) and nearest-
    upsample instead, which is ~1.6s regardless of block size since it's a
    reshape+min, not a sliding window. Blocky (piecewise-constant) output is
    fine here - we only ever sample this at line/polygon vertices, not draw
    it as a continuous terrain surface."""
    block = max(1, round(GROUND_HEIGHT_BLOCK_M / resolution))
    h, w = height.shape
    pad_h, pad_w = (-h) % block, (-w) % block
    padded = np.pad(height, ((0, pad_h), (0, pad_w)), mode="edge")
    blocks = padded.reshape(padded.shape[0] // block, block, padded.shape[1] // block, block)
    coarse = blocks.min(axis=(1, 3))
    upsampled = np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)
    return upsampled[:h, :w]


def load_grids(scene_dir: Path, mask_path: Path):
    meta = json.loads((scene_dir / "meta.json").read_text())
    mask = tifffile.imread(mask_path)
    height = tifffile.imread(scene_dir / "height.tif").astype(np.float64)
    intensity = tifffile.imread(scene_dir / "intensity.tif")
    valid = intensity != -9999

    xmin, ymin, xmax, ymax = meta["bounds"]
    resolution = meta["resolution"]
    transform = Affine(resolution, 0, xmin, 0, -resolution, ymax)

    # nearest-fill height so pixels the model marked as line/crosswalk but
    # that happen to sit on a no-LiDAR-coverage cell still get a plausible Z
    # instead of falling back to a single flat height for the whole layer.
    if not valid.all():
        _, (nr, nc) = ndimage.distance_transform_edt(~valid, return_indices=True)
        height = height[nr, nc]

    height = _robust_ground_height(height, resolution)

    return mask, height, valid, transform, resolution, meta.get("epsg", "EPSG:32652")


def _remove_skeleton_branch_points(skeleton: np.ndarray) -> np.ndarray:
    """lane_line은 중앙선/차선/유도선 등 여러 종류가 한 클래스로 합쳐져 있어서,
    교차로 근처에서 서로 다른 선들이 dilation 후 하나의 덩어리로 붙어버리는 경우가
    흔하다 - 그 위에서 스켈레톤을 뽑으면 갈래(분기)가 있는 스켈레톤이 나오는데,
    성분 전체를 주성분(PCA) 축 하나로 정렬해 한 줄로 펴려고 하면 갈래를 왔다갔다
    하면서 지그재그("worm")가 된다 (실측: 교차로 부근 lane 라인이 실제로 이렇게
    나왔음). 분기점(8-이웃 3개 이상)을 스켈레톤에서 제거하면 각 갈래가 별도의
    단순한(분기 없는) 성분으로 쪼개져 PCA 정렬이 제대로 먹힌다 - 분기점 제거로
    생기는 작은 틈은 이미 있는 대시(점선) 이어붙이기 로직이 방향이 맞으면 다시
    이어주고, 진짜 갈라지는 지점(교차로에서 직진/회전차선이 갈라지는 곳 등)은
    방향이 안 맞아 안 이어져서 별개의 선으로 남는다 - 오히려 원하는 동작."""
    neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8),
                                       mode="constant") - skeleton
    branch_points = skeleton & (neighbor_count >= 3)
    return skeleton & ~branch_points


def _extract_skeleton_segments(binary: np.ndarray, z_grid: np.ndarray, transform: Affine,
                                resolution: float, min_length_m: float):
    """대시(점선) 하나하나가 스켈레톤 연결요소 하나가 됨 - 주성분(PCA) 축으로
    픽셀 순서를 매기고, Savitzky-Golay로 픽셀 단위 지그재그를 다듬어서 세그먼트로
    반환 (뒤에서 대시끼리 이어붙임)."""
    closed = ndimage.binary_closing(binary, structure=disk(max(1, round(0.3 / resolution))))
    skeleton = _remove_skeleton_branch_points(skeletonize(closed))

    labeled, _ = ndimage.label(skeleton, structure=np.ones((3, 3)))
    objects = ndimage.find_objects(labeled)
    min_cells = round(min_length_m / resolution)
    seg_window = _odd_window(SEGMENT_SMOOTH_WINDOW_M, resolution, 5)

    segments = []
    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        sub = labeled[slc] == comp_id
        if sub.sum() < min_cells:
            continue
        rows_local, cols_local = np.where(sub)
        rows = rows_local + slc[0].start
        cols = cols_local + slc[1].start

        pts = np.column_stack([cols, rows]).astype(float)
        centered = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        axis = vt[0]
        order = np.argsort(centered @ axis)

        rows_ord = rows[order].astype(float)
        cols_ord = cols[order].astype(float)
        zs = z_grid[rows[order], cols[order]]

        if len(rows_ord) >= seg_window:
            rows_ord = savgol_filter(rows_ord, seg_window, 2, mode="nearest")
            cols_ord = savgol_filter(cols_ord, seg_window, 2, mode="nearest")

        xs, ys = transform * (cols_ord, rows_ord)
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
        cos_angle = float(np.dot(di, dj))
        if cos_angle > -max_angle_cos:
            continue
        candidates.append((float(np.linalg.norm(pi - pj)), i, ei, j, ej))
    candidates.sort(key=lambda c: c[0])

    port_link = {}
    for _, i, ei, j, ej in candidates:
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


def _fit_chain(segments, chain, resolution):
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
    order = np.argsort(centered @ axis)

    residual_rms = float(np.sqrt(np.mean((centered @ vt[1]) ** 2))) if vt.shape[0] > 1 else 0.0

    if residual_rms <= STRAIGHT_FIT_RESIDUAL_TOL:
        i0, i1 = order[0], order[-1]
        return LineString([tuple(xy[i0]) + (float(z[i0]),), tuple(xy[i1]) + (float(z[i1]),)])

    xy_ord, z_ord = xy[order].copy(), z[order]
    chain_window = _odd_window(CHAIN_SMOOTH_WINDOW_M, resolution, 7)
    if len(xy_ord) >= chain_window:
        xy_ord[:, 0] = savgol_filter(xy_ord[:, 0], chain_window, 2, mode="nearest")
        xy_ord[:, 1] = savgol_filter(xy_ord[:, 1], chain_window, 2, mode="nearest")
    line_2d = LineString(list(zip(xy_ord[:, 0].tolist(), xy_ord[:, 1].tolist()))).simplify(LANE_SIMPLIFY_TOL)
    if line_2d.is_empty or line_2d.length == 0:
        return None
    nn_tree = cKDTree(xy_ord)
    _, nn_idx = nn_tree.query(np.array(line_2d.coords))
    return LineString([(x, y, float(z_ord[i])) for (x, y), i in zip(line_2d.coords, nn_idx)])


def extract_lines(binary: np.ndarray, z_grid: np.ndarray, transform: Affine,
                   resolution: float, min_length_m: float) -> list[LineString]:
    segments = _extract_skeleton_segments(binary, z_grid, transform, resolution, min_length_m)
    chains = _stitch_segments(segments, max_gap=DASH_MAX_GAP_M, max_angle_deg=DASH_MAX_ANGLE_DEG)
    chains = _split_bendy_chains(segments, chains, max_total_bend_deg=DASH_MAX_TOTAL_BEND_DEG)
    lines = [_fit_chain(segments, chain, resolution) for chain in chains]
    return [l for l in lines if l is not None]


def extract_polygons(binary: np.ndarray, z_grid: np.ndarray, transform: Affine,
                      resolution: float, min_area_m2: float) -> list[Polygon]:
    closed = ndimage.binary_closing(binary, structure=np.ones((3, 3)))
    labeled, _ = ndimage.label(closed, structure=np.ones((3, 3)))
    objects = ndimage.find_objects(labeled)
    min_cells = round(min_area_m2 / (resolution ** 2))

    polygons = []
    for comp_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        sub_transform = transform * Affine.translation(slc[1].start, slc[0].start)
        mask = labeled[slc] == comp_id
        if mask.sum() < min_cells:
            continue
        for geom, value in rasterio.features.shapes(mask.astype(np.uint8), mask=mask, transform=sub_transform):
            if value != 1:
                continue
            poly = shapely_shape(geom).simplify(CROSSWALK_SIMPLIFY_TOL)
            if poly.is_empty or poly.area == 0:
                continue

            def ring_with_z(coords):
                out = []
                for x, y in coords:
                    col, row = ~transform * (x, y)
                    col = int(np.clip(col, 0, z_grid.shape[1] - 1))
                    row = int(np.clip(row, 0, z_grid.shape[0] - 1))
                    out.append((x, y, float(z_grid[row, col])))
                return out

            exterior = ring_with_z(poly.exterior.coords)
            interiors = [ring_with_z(r.coords) for r in poly.interiors]
            polygons.append(Polygon(exterior, interiors))
    return polygons


def _map_z(coords, fn):
    """coords is a GeoJSON coordinates array - (x,y,z) for a point, or
    arbitrarily nested sequences of those for LineString/Polygon rings
    (shapely's mapping() returns tuples, so this returns a new structure
    rather than mutating in place). Applies fn to every triple's z."""
    if isinstance(coords[0], (int, float)):
        return (coords[0], coords[1], fn(coords[2]))
    return [_map_z(c, fn) for c in coords]


def _collect_z(coords, out):
    if isinstance(coords[0], (int, float)):
        out.append(coords[2])
    else:
        for c in coords:
            _collect_z(c, out)


def clip_class_z_outliers(geometries: list[dict], mad_k: float = 3.0) -> None:
    """block-min in _robust_ground_height only escapes contamination that's
    smaller than its block - confirmed on real data that a contaminated patch
    can span 10s of meters (a whole cluster of output vertices shared one
    identical Z far above everything else nearby, i.e. no clean ground return
    existed anywhere in the block-min's neighborhood there either). A global
    per-scene percentile doesn't help either - the scan bounds include real
    tall things (buildings, trees) at legitimately high Z, which loosens the
    percentile past the actual contamination. And even scoped to one class's
    own geometry, a plain [1,99] percentile still isn't robust enough - on
    real data the contaminated cluster was ~1.17% of that class's vertices,
    which sat the 99th percentile itself right on top of the contamination.
    Median absolute deviation is robust to outlier fractions far higher than
    that (breakdown point 50%), so use median +/- mad_k*1.4826*MAD instead -
    confirmed this actually separates a contamination cluster that size."""
    all_z = []
    for g in geometries:
        _collect_z(g["coordinates"], all_z)
    if len(all_z) < 10:
        return
    all_z = np.array(all_z)
    median = np.median(all_z)
    mad = np.median(np.abs(all_z - median))
    if mad == 0:
        return
    robust_std = 1.4826 * mad
    lo, hi = median - mad_k * robust_std, median + mad_k * robust_std
    clip_fn = lambda z: float(np.clip(z, lo, hi))
    for g in geometries:
        g["coordinates"] = _map_z(g["coordinates"], clip_fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, required=True, help="ml/rasterize.py 출력 디렉토리 (height.tif, intensity.tif, meta.json)")
    ap.add_argument("--mask", type=Path, required=True, help="클래스 id 래스터 (ml/infer.py --out-mask 결과)")
    ap.add_argument("--out", type=Path, required=True, help="출력 GeoJSON 경로")
    args = ap.parse_args()

    mask, height, valid, transform, resolution, epsg = load_grids(args.scene_dir, args.mask)
    print(f"mask shape {mask.shape}, resolution {resolution}m, epsg {epsg}")

    features = []
    for class_id, (name, kind) in CLASS_NAME_TO_GEOM.items():
        # gt/pred rasters both cover the full LAS-bounds grid regardless of
        # actual scan coverage - a no-coverage cell has no real intensity/
        # height signal (fed as zeros to the model), so any class the model
        # "predicts" there is spurious, not a real detection (confirmed: raw
        # lane px count without this filter was 10.9M vs 845K with it, on
        # the same array - the difference was all outside LiDAR coverage).
        binary = (mask == class_id) & valid
        if not binary.any():
            print(f"  {name}: no predicted pixels, skipping")
            continue
        if kind == "line":
            geoms = extract_lines(binary, height, transform, resolution, MIN_LANE_LENGTH_M)
            print(f"  {name}: {int(binary.sum())} px -> {len(geoms)} lines")
        else:
            geoms = extract_polygons(binary, height, transform, resolution, MIN_CROSSWALK_AREA_M2)
            print(f"  {name}: {int(binary.sum())} px -> {len(geoms)} polygons")
        mapped = [mapping(g) for g in geoms]
        clip_class_z_outliers(mapped)
        for m in mapped:
            features.append({"type": "Feature", "properties": {"class": name}, "geometry": m})

    if not features:
        raise SystemExit("no vector features extracted - nothing to write")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    print(f"Done -> {args.out} ({len(features)} features)")


if __name__ == "__main__":
    main()
