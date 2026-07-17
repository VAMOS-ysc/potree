#!/usr/bin/env python3
"""
LAS/LAZ 포인트클라우드와 정밀도로지도 SHP(B2_SURFACELINEMARK, B3_SURFACEMARK)를
같은 좌표계·해상도의 정렬된 2D 래스터(BEV, bird's-eye-view)로 변환한다.
U-Net 등 세그멘테이션 모델 학습용 (입력 이미지, 정답 마스크) 쌍을 만드는
전처리 스크립트.

의존성: PDAL, GDAL(gdal_rasterize/gdalsrsinfo)이 PATH에 있어야 함 (README
Prerequisites 참고 - webapp/server.py가 이미 이 도구들에 의존하므로 Lane
Digitize Tool이 동작한다면 이미 설치되어 있다). Python 패키지는
ml/requirements.txt 참고.

출력 (--out DIR 아래):
    intensity.tif   - LAS Intensity 평균값 래스터 (float32, 1 band)
    height.tif      - LAS Z 평균값 래스터 (float32, 1 band)
    mask.tif        - 클래스 ID 래스터 (uint8, 1 band). 클래스는 CLASS_NAMES 참고
    mask_preview.png - mask.tif를 색으로 눈으로 확인하기 위한 미리보기 (선택)
    meta.json       - bounds/resolution/epsg/class map 기록

SHP가 LAS와 다른 좌표계면 (예: 정밀도로지도 납품 SHP는 흔히 EPSG:4326 경위도,
B3_SURFACEMARK처럼 .prj가 아예 없는 경우도 있음) --las의 좌표계로 자동 재투영한다.
출력 영역은 --las가 있으면 그 점군의 bounds로 고정된다 (SHP가 도시 전역을
담고 있어도 점군이 커버하는 범위만 잘라서 씀) - --las 없이 SHP만 줄 때만
SHP들의 합집합 영역을 쓴다.

사용 예:
    python ml/rasterize.py \\
        --las path/to/08-001.las \\
        --lines path/to/B2_SURFACELINEMARK.shp \\
        --crosswalks path/to/B3_SURFACEMARK.shp \\
        --out out/08-001 \\
        --resolution 0.05 \\
        --preview
"""
import argparse
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import binary_dilation

# B2_SURFACELINEMARK의 Kind(선규제유형) 코드 -> 학습 클래스 ID.
# 코드 정의 출처: 정밀도로지도 제작 매뉴얼(2020.12) 9.3.7절 <표 9.52>.
# --class-map 옵션으로 이 파일 없이 JSON을 넘겨서 덮어쓸 수 있다.
#
# 차선/정지선을 예전엔 클래스 1/2로 나눴는데, 카메라가 아니라 LAS(라이다)만 보는
# 이상 도로에 칠해진 흰 페인트 선이라는 점에서 구분할 근거가 약해서 (2026-07-16
# 요청) line_marking 하나로 합침 - 530(정지선), 515(주정차금지선, 국토지리정보원
# 공식 지도에서 처음 봄), 5011(가변차선, 501의 하위코드)도 전부 여기 포함.
# 531(안전지대)/599(기타선)는 여전히 별도(other) - 페인트 선이라기보단 구역
# 표시/미정의 유형이라 성격이 달라서 그대로 둠.
DEFAULT_KIND_TO_CLASS = {
    "501": 1,    # 중앙선
    "5011": 1,   # 가변차선 (501의 하위코드)
    "502": 1,    # 유턴구역선
    "503": 1,    # 차선
    "504": 1,    # 버스전용차선
    "505": 1,    # 길가장자리구역선
    "506": 1,    # 진로변경제한선
    "515": 1,    # 주정차금지선
    "525": 1,    # 유도선
    "530": 1,    # 정지선
    "535": 1,    # 자전거도로
    "531": 3,    # 안전지대
    "599": 3,    # 기타선
}
# B3_SURFACEMARK은 Type(표시형태)=5가 횡단보도 카테고리 (9.3.8절 <표 9.55>), 그 안에서
# Kind(표시종류, <표 9.56>)로 다시 나뉜다: 5321=일반 횡단보도, 533=고원식횡단보도,
# 534=자전거횡단보도, 524=정차금지대(간혹 잘못 섞여 들어옴). 학습에는 일반 횡단보도
# (5321)만 쓰기로 함 (2026-07-15 요청) - 고원식/자전거 횡단보도는 제외.
CROSSWALK_TYPE_VALUE = "5"
CROSSWALK_KIND_VALUE = "5321"
CROSSWALK_CLASS = 2

CLASS_NAMES = {
    0: "background",
    1: "line_marking",
    2: "crosswalk",
    3: "other",
}

PREVIEW_COLORS = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (60, 140, 255),
    3: (120, 120, 120),
}


def run(cmd, **kwargs):
    result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr}")
    return result


def las_bounds_epsg(las_path: Path):
    summary = json.loads(run(["pdal", "info", "--summary", str(las_path)]).stdout)["summary"]
    b = summary["bounds"]
    # gdalsrsinfo can't read LAS directly, and PDAL's srs.json.components field isn't
    # populated on every PDAL build - pull the EPSG code out of the WKT text instead,
    # taking the last AUTHORITY match (the outer PROJCS/GEOGCS, not a nested DATUM one).
    wkt = summary.get("srs", {}).get("horizontal") or summary.get("srs", {}).get("compoundwkt") or ""
    matches = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
    epsg = f"EPSG:{matches[-1]}" if matches else None
    return (b["minx"], b["miny"], b["maxx"], b["maxy"]), epsg


def shp_bounds(shp_path: Path):
    out = run(["ogrinfo", "-al", "-so", str(shp_path)]).stdout
    m = re.search(
        r"Extent:\s*\(([-\d.]+),\s*([-\d.]+)\)\s*-\s*\(([-\d.]+),\s*([-\d.]+)\)", out
    )
    if not m:
        raise RuntimeError(f"could not parse extent from ogrinfo output for {shp_path}")
    return tuple(float(x) for x in m.groups())


def shp_epsg(shp_path: Path) -> str | None:
    result = subprocess.run(["gdalsrsinfo", "-o", "epsg", str(shp_path)], capture_output=True, text=True)
    return result.stdout.strip() or None


def normalize_shp(shp_path: Path, target_epsg: str, assumed_src_epsg: str, tmp_dir: Path) -> Path:
    """SHP를 target_epsg로 재투영한 사본을 tmp_dir에 만들어 반환한다.

    B3_SURFACEMARK처럼 .prj가 없어 SRS를 판별할 수 없는 레이어는 흔히 정밀도로지도
    납품 스펙대로 EPSG:4326(경위도)라고 가정하고 assumed_src_epsg를 원본 좌표계로
    사용한다 (--shp-srs로 조정 가능). 이미 target_epsg인 레이어도 그대로 통과시킨다
    (identity 변환, 약간 비효율적이지만 로직이 단순해짐).
    """
    detected = shp_epsg(shp_path)
    out_path = tmp_dir / f"{shp_path.stem}_{target_epsg.replace(':', '')}.shp"
    cmd = ["ogr2ogr", "-t_srs", target_epsg]
    if not detected:
        print(f"  {shp_path.name}: no CRS defined, assuming {assumed_src_epsg}", file=sys.stderr)
        cmd += ["-s_srs", assumed_src_epsg]
    cmd += [str(out_path), str(shp_path)]
    run(cmd)
    return out_path


def union_bounds(bounds_list):
    xs_min = [b[0] for b in bounds_list]
    ys_min = [b[1] for b in bounds_list]
    xs_max = [b[2] for b in bounds_list]
    ys_max = [b[3] for b in bounds_list]
    return (min(xs_min), min(ys_min), max(xs_max), max(ys_max))


def raster_size(bounds, resolution):
    # Must match PDAL writers.gdal's own width/height formula (ceil, not round) -
    # otherwise mask.tif ends up a pixel off from intensity.tif/height.tif on edges
    # where (extent / resolution) isn't a whole number, which is the common case.
    xmin, ymin, xmax, ymax = bounds
    width = max(1, math.ceil((xmax - xmin) / resolution))
    height = max(1, math.ceil((ymax - ymin) / resolution))
    return width, height


def tif_size(tif_path: Path):
    with tifffile.TiffFile(tif_path) as tf:
        page = tf.pages[0]
        return page.imagewidth, page.imagelength


def rasterize_las_dimension(las_path: Path, dimension: str, bounds, resolution: float, out_tif: Path):
    # NB: tried branching one readers.las into two writers.gdal (tag/inputs) to
    # rasterize Intensity+Z in a single pass over multi-GB files - PDAL 2.3.0
    # silently drops the second writer (no error, file just never gets written).
    # Two separate passes cost more I/O but actually produce both outputs.
    xmin, ymin, xmax, ymax = bounds
    pipeline = {
        "pipeline": [
            {"type": "readers.las", "filename": str(las_path)},
            {
                "type": "writers.gdal",
                "filename": str(out_tif),
                "resolution": resolution,
                "output_type": "mean",
                "dimension": dimension,
                "data_type": "float32",
                "nodata": -9999,
                "bounds": f"([{xmin},{xmax}],[{ymin},{ymax}])",
            },
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(pipeline, f)
        pipeline_path = f.name
    run(["pdal", "pipeline", pipeline_path])


def init_mask(bounds, resolution, epsg, out_tif: Path, size=None):
    # size, when given, comes from an actual already-rasterized intensity.tif/height.tif
    # (see main()) - always prefer that over recomputing from bounds/resolution, since
    # it's the ground truth for what those files' grid actually is.
    width, height = size if size else raster_size(bounds, resolution)
    xmin, ymin, xmax, ymax = bounds
    args = [
        "gdal_create",
        "-ot", "Byte",
        "-outsize", str(width), str(height),
        "-a_srs", epsg or "EPSG:4326",
        "-a_ullr", str(xmin), str(ymax), str(xmax), str(ymin),
        "-burn", "0",
        str(out_tif),
    ]
    run(args)


def burn_where(shp_path: Path, where: str, class_id: int, out_tif: Path):
    run([
        "gdal_rasterize",
        "-b", "1",
        "-burn", str(class_id),
        "-where", where,
        str(shp_path), str(out_tif),
    ])


def burn_classes(shp_path: Path, field: str, value_to_class: dict, out_tif: Path):
    for value, class_id in value_to_class.items():
        burn_where(shp_path, f"{field} = '{value}'", class_id, out_tif)


def disk_structure(radius_px: int):
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x ** 2 + y ** 2) <= radius_px ** 2


def widen_thin_classes(mask: np.ndarray, resolution: float, widths_m: dict) -> np.ndarray:
    """B2_SURFACELINEMARK is a bare LineString layer (no width) - burned as-is it
    rasterizes to a ~1px-wide skeleton (confirmed empirically: median cross-section
    run length was exactly 1px on real data), which is both nearly unlearnable
    through a downsampling U-Net and unfairly punished by IoU on any sub-pixel
    misalignment. Real paint has width (~15cm lane lines, ~40cm Korean stop lines) -
    dilate each thin class to that footprint before training instead.

    widths_m: {class_id: width_in_meters}, processed in the same low->high priority
    order the original sequential gdal_rasterize burns used (lane_line, stop_line,
    other), so overlaps resolve the same way as before, just wider. Crosswalk is
    already an area polygon (not in widths_m) and always wins at overlaps, matching
    the original burn order where it's rasterized last.
    """
    result = mask.copy()
    crosswalk_px = mask == CROSSWALK_CLASS
    for class_id, width_m in widths_m.items():
        radius_px = round((width_m / resolution) / 2)
        if radius_px <= 0:
            result[mask == class_id] = class_id
            continue
        dilated = binary_dilation(mask == class_id, structure=disk_structure(radius_px))
        result[dilated] = class_id
    result[crosswalk_px] = CROSSWALK_CLASS
    return result


def write_preview(mask_tif: Path, out_png: Path):
    mask = tifffile.imread(mask_tif)
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in PREVIEW_COLORS.items():
        rgb[mask == class_id] = color
    Image.fromarray(rgb).save(out_png)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--las", type=Path, help="입력 LAS/LAZ 파일")
    ap.add_argument("--lines", type=Path, help="B2_SURFACELINEMARK.shp (차선/정지선 등)")
    ap.add_argument("--crosswalks", type=Path, help="B3_SURFACEMARK.shp (횡단보도 등)")
    ap.add_argument("--out", type=Path, required=True, help="출력 디렉토리")
    ap.add_argument("--resolution", type=float, default=0.05, help="셀 크기 (m/pixel), 기본 0.05")
    ap.add_argument("--bounds", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                     help="명시적 영역 (안 주면 입력 파일들의 합집합 영역을 사용)")
    ap.add_argument("--class-map", type=Path, help="DEFAULT_KIND_TO_CLASS를 덮어쓸 JSON 파일")
    ap.add_argument("--preview", action="store_true", help="mask.tif의 색상 미리보기 PNG도 생성")
    ap.add_argument("--epsg", help="목표 좌표계 (기본: --las의 좌표계). SHP는 이 좌표계로 재투영된다")
    ap.add_argument("--shp-srs", default="EPSG:4326",
                     help="SRS(.prj)가 없는 SHP의 원본 좌표계로 가정할 값 (기본 EPSG:4326 - "
                          "정밀도로지도 납품 스펙의 경위도. 예: B3_SURFACEMARK는 보통 .prj가 없음)")
    ap.add_argument("--lane-width", type=float, default=0.15,
                     help="line_marking/other 클래스를 이 폭(m)만큼 넓혀서 마스크에 굽는다 - "
                          "B2_SURFACELINEMARK는 폭 없는 LineString이라 그대로 구우면 ~1px 선이 되어 "
                          "학습도 안 되고 IoU도 사소한 오차에 0이 됨 (기본 0.15m, 실제 차선 폭 근사 - "
                          "정지선은 실제로 더 두껍지만 2026-07-16부터 차선과 같은 클래스로 합쳐서 "
                          "굳이 따로 폭을 안 줌)")
    ap.add_argument("--mask-only", action="store_true",
                     help="intensity.tif/height.tif는 이미 --out에 있다고 보고 다시 굽지 않음 - "
                          "마스크 폭 등 라벨 관련 파라미터만 바꿔서 빠르게 재생성할 때 사용 (LAS 재읽기 생략)")
    args = ap.parse_args()

    if args.mask_only:
        existing_meta_path = args.out / "meta.json"
        if not existing_meta_path.exists():
            ap.error(f"--mask-only는 {existing_meta_path}가 이미 있어야 함 (bounds/epsg를 거기서 읽음)")
        if not (args.out / "intensity.tif").exists():
            ap.error(f"--mask-only는 {args.out / 'intensity.tif'}가 이미 있어야 함 (마스크 크기를 거기서 읽음)")
    elif not (args.las or args.lines or args.crosswalks or args.bounds):
        ap.error("--las, --lines, --crosswalks 중 최소 하나 또는 --bounds가 필요합니다")

    kind_to_class = DEFAULT_KIND_TO_CLASS
    if args.class_map:
        kind_to_class = json.loads(args.class_map.read_text())

    epsg = args.epsg
    las_bounds = None
    if args.mask_only:
        existing_meta = json.loads((args.out / "meta.json").read_text())
        epsg = epsg or existing_meta["epsg"]
        las_bounds = tuple(existing_meta["bounds"])
    elif args.las:
        las_bounds, las_epsg = las_bounds_epsg(args.las)
        epsg = epsg or las_epsg
    if (args.lines or args.crosswalks) and not epsg:
        ap.error("SHP를 재투영할 목표 좌표계를 알 수 없습니다: --las를 주거나 --epsg를 직접 지정하세요")

    args.out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as norm_dir_str:
        norm_dir = Path(norm_dir_str)
        lines_shp = normalize_shp(args.lines, epsg, args.shp_srs, norm_dir) if args.lines else None
        crosswalks_shp = normalize_shp(args.crosswalks, epsg, args.shp_srs, norm_dir) if args.crosswalks else None

        if args.bounds:
            bounds = tuple(args.bounds)
        elif las_bounds:
            # The point cloud defines the study area. SHP layers (e.g. citywide
            # 정밀도로지도 deliverables) are usually far bigger than one scan's
            # coverage - unioning with them would blow the raster up to city size.
            # gdal_rasterize just clips to whatever's burned onto this extent.
            bounds = las_bounds
        else:
            bounds_candidates = []
            if lines_shp:
                bounds_candidates.append(shp_bounds(lines_shp))
            if crosswalks_shp:
                bounds_candidates.append(shp_bounds(crosswalks_shp))
            bounds = union_bounds(bounds_candidates)

        if args.las and not args.mask_only:
            print("Rasterizing intensity...", file=sys.stderr)
            rasterize_las_dimension(args.las, "Intensity", bounds, args.resolution, args.out / "intensity.tif")
            print("Rasterizing height (Z)...", file=sys.stderr)
            rasterize_las_dimension(args.las, "Z", bounds, args.resolution, args.out / "height.tif")

        mask_tif = args.out / "mask.tif"
        if lines_shp or crosswalks_shp:
            print("Building class mask...", file=sys.stderr)
            has_intensity = args.mask_only or (args.out / "intensity.tif").exists()
            las_size = tif_size(args.out / "intensity.tif") if has_intensity else None
            init_mask(bounds, args.resolution, epsg, mask_tif, size=las_size)
            if lines_shp:
                burn_classes(lines_shp, "Kind", kind_to_class, mask_tif)
            if crosswalks_shp:
                burn_where(
                    crosswalks_shp,
                    f"Type = '{CROSSWALK_TYPE_VALUE}' AND Kind = '{CROSSWALK_KIND_VALUE}'",
                    CROSSWALK_CLASS, mask_tif,
                )
            print("Widening thin line classes...", file=sys.stderr)
            thin_widths = {}
            for class_id in dict.fromkeys(kind_to_class.values()):  # first-occurrence order, matches burn order
                if class_id == CROSSWALK_CLASS:
                    continue
                thin_widths[class_id] = args.lane_width
            mask = tifffile.imread(mask_tif)
            mask = widen_thin_classes(mask, args.resolution, thin_widths)
            tifffile.imwrite(mask_tif, mask)
            if args.preview:
                write_preview(mask_tif, args.out / "mask_preview.png")

    meta = {
        "bounds": bounds,
        "resolution": args.resolution,
        "epsg": epsg,
        "class_names": CLASS_NAMES,
        "kind_to_class": kind_to_class,
        "source": {
            "las": str(args.las) if args.las else None,
            "lines": str(args.lines) if args.lines else None,
            "crosswalks": str(args.crosswalks) if args.crosswalks else None,
        },
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Done -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
