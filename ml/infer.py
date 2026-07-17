#!/usr/bin/env python3
"""학습된 U-Net 체크포인트로 한 씬 전체(intensity.tif/height.tif)를 추론하고,
결과를 potree에서 바로 볼 수 있는 포인트 기반 GeoJSON으로 내보낸다.

examples/test_predictions.html?dataset=<potree dataset>&predictions=<이 GeoJSON 경로>
로 열면 포인트클라우드 위에 예측된 차선/정지선/횡단보도가 겹쳐 보인다.

각 클래스의 예측 픽셀 좌표(X, Y)는 meta.json의 bounds/resolution으로 월드 좌표로
변환하고, Z는 같은 씬의 height.tif(포인트클라우드 평균 높이)에서 그대로 읽어 쓴다 -
래스터 기반 예측이라 원래 Z가 없지만, 이렇게 하면 실제 지형 높이에 맞게 찍힌다.

사용 예:
    python ml/infer.py \\
        --scene-dir out/09-002 \\
        --checkpoint checkpoints_v2/best.pt \\
        --config checkpoints_v2/config.json \\
        --out ../pointclouds/09-002_.../predictions_v2.geojson
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch

from dataset import load_scene, _tile_positions
from unet import UNet

# rasterize.py's CLASS_NAMES id -> the class name test_predictions.html's viewer
# groups points by (byClass = {lane, crosswalk, ...}). "other"(4)/background(0)
# are skipped - too rare / not meaningful to plot as points.
CLASS_TO_GEOJSON_NAME = {1: "lane", 2: "crosswalk"}
MAX_POINTS_PER_CLASS = 80_000


def tiled_predict(model, scene, patch_size: int, device) -> np.ndarray:
    h, w = scene.mask.shape
    pred = np.zeros((h, w), dtype=np.uint8)
    ys = _tile_positions(h, patch_size, patch_size)
    xs = _tile_positions(w, patch_size, patch_size)
    model.eval()
    total = len(ys) * len(xs)
    done = 0
    with torch.no_grad():
        for y in ys:
            for x in xs:
                sl = (slice(y, y + patch_size), slice(x, x + patch_size))
                intensity = (scene.intensity[sl] - scene.intensity_mean) / scene.intensity_std
                height = (scene.height[sl] - scene.height_mean) / scene.height_std
                valid = scene.valid[sl]
                intensity = np.where(valid, intensity, 0.0).astype(np.float32)
                height = np.where(valid, height, 0.0).astype(np.float32)
                inp = torch.from_numpy(np.stack([intensity, height], axis=0)[None]).to(device)
                logits = model(inp)
                pred[sl] = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"  predicted {done}/{total} patches", file=sys.stderr)
    return pred


def mask_to_geojson(pred: np.ndarray, height_raster: np.ndarray, valid: np.ndarray,
                     bounds, resolution: float) -> dict:
    xmin, ymin, xmax, ymax = bounds
    features = []
    rng = np.random.default_rng(0)
    for class_id, name in CLASS_TO_GEOJSON_NAME.items():
        ys, xs = np.where((pred == class_id) & valid)
        n = len(ys)
        if n == 0:
            continue
        if n > MAX_POINTS_PER_CLASS:
            idx = rng.choice(n, size=MAX_POINTS_PER_CLASS, replace=False)
            ys, xs = ys[idx], xs[idx]
        world_x = xmin + (xs.astype(np.float64) + 0.5) * resolution
        world_y = ymax - (ys.astype(np.float64) + 0.5) * resolution
        world_z = height_raster[ys, xs].astype(np.float64)
        print(f"  class {name}: {n} predicted px -> {len(ys)} points in geojson", file=sys.stderr)
        for wx, wy, wz in zip(world_x, world_y, world_z):
            features.append({
                "type": "Feature",
                "properties": {"class": name},
                "geometry": {"type": "Point", "coordinates": [wx, wy, wz]},
            })
    return {"type": "FeatureCollection", "features": features}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, required=True, help="ml/rasterize.py 출력 디렉토리 (intensity/height/mask.tif, meta.json)")
    ap.add_argument("--checkpoint", type=Path, required=True, help="train.py가 저장한 .pt 체크포인트")
    ap.add_argument("--config", type=Path, required=True, help="train.py가 같이 저장한 config.json")
    ap.add_argument("--out", type=Path, required=True, help="출력 GeoJSON 경로 (포인트 스플래터, 빠른 확인용)")
    ap.add_argument("--out-mask", type=Path, help="예측된 클래스 id 래스터(mask.tif와 같은 형식)도 저장할 경로 - "
                     "vectorize_predictions.py가 포인트 재격자화 없이 이 래스터를 직접 스켈레톤화/폴리곤화함")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    meta = json.loads((args.scene_dir / "meta.json").read_text())

    print(f"Loading scene {args.scene_dir}...", file=sys.stderr)
    scene = load_scene(args.scene_dir, require_mask=False)

    device = torch.device(args.device)
    model = UNet(in_channels=config["in_channels"], num_classes=len(config["class_names"]),
                 base_channels=config["base_channels"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    print("Running tiled inference...", file=sys.stderr)
    pred = tiled_predict(model, scene, config["patch_size"], device)

    if args.out_mask:
        args.out_mask.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(args.out_mask, pred)
        print(f"Wrote raw prediction raster -> {args.out_mask}", file=sys.stderr)

    geojson = mask_to_geojson(pred, scene.height, scene.valid, meta["bounds"], meta["resolution"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(geojson))
    print(f"Done -> {args.out} ({len(geojson['features'])} points)", file=sys.stderr)


if __name__ == "__main__":
    main()
