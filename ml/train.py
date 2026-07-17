#!/usr/bin/env python3
"""U-Net 학습 스크립트. ml/rasterize.py로 만든 씬(intensity.tif/height.tif/mask.tif)
여러 개를 patch로 잘라 차선/횡단보도/정지선 세그멘테이션 모델을 학습한다.

씬 단위로 train/val을 나눈다 (같은 도로에서 나온 인접 patch가 train/val에 같이
섞여 val 점수가 부풀려지는 걸 막기 위해) - 데이터가 씬 하나뿐일 때만 예외적으로
patch 단위 랜덤 분할로 대체한다 (경고 출력).

GPU 전력: 이 머신은 PSU 960W 한도 때문에 GPU 파워리밋이 이미 220W로 걸려 있음
(gpu-power-limit.service) - 학습 스크립트가 따로 조절할 필요 없음, 드라이버가
알아서 클럭을 낮춰서 220W를 넘기지 않는다. 배치 크기를 무리하게 키우지만 않으면 됨.

사용 예:
    # 씬 디렉토리 구조: ml/out/<scene_name>/{intensity.tif,height.tif,mask.tif,meta.json}
    python ml/train.py \\
        --data-root ml/out \\
        --val-scenes 08-001 04 \\
        --patch-size 256 --batch-size 8 --epochs 30 \\
        --out ml/checkpoints
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import IGNORE_INDEX, PatchDataset, build_patch_index, class_weights, load_scenes
from rasterize import CLASS_NAMES
from unet import UNet

NUM_CLASSES = len(CLASS_NAMES)


def dice_loss(logits, target, num_classes, ignore_index, eps=1e-6):
    valid = target != ignore_index
    target_safe = target.clone()
    target_safe[~valid] = 0
    probs = F.softmax(logits, dim=1)
    target_onehot = F.one_hot(target_safe, num_classes).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    target_onehot = target_onehot * valid_f
    dims = (0, 2, 3)
    intersection = (probs * target_onehot).sum(dims)
    union = probs.sum(dims) + target_onehot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def confusion_matrix(preds, targets, num_classes, ignore_index):
    valid = targets != ignore_index
    preds, targets = preds[valid], targets[valid]
    idx = targets * num_classes + preds
    return torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def iou_per_class(conf):
    diag = conf.diag().float()
    denom = conf.sum(0).float() + conf.sum(1).float() - diag
    return torch.where(denom > 0, diag / denom, torch.full_like(diag, float("nan")))


def discover_scenes(data_root: Path):
    return sorted(d for d in data_root.iterdir() if d.is_dir() and (d / "mask.tif").exists())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, help="씬 디렉토리들의 부모 디렉토리")
    ap.add_argument("--scenes", type=Path, nargs="+", help="--data-root 대신 씬 디렉토리를 직접 나열")
    ap.add_argument("--val-scenes", nargs="*", default=[], help="검증용으로 뺄 씬 이름들 (씬 디렉토리 이름)")
    ap.add_argument("--patch-size", type=int, default=256, help="16의 배수 권장 (다운샘플 4단계)")
    ap.add_argument("--stride", type=int, default=None, help="기본값 = patch-size (겹침 없음)")
    ap.add_argument("--min-valid-frac", type=float, default=0.1, help="이보다 LiDAR 커버리지 낮은 패치는 버림")
    ap.add_argument("--bg-keep-ratio", type=float, default=0.3, help="라벨 없는(배경만) 패치 중 남길 비율")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--base-channels", type=int, default=32)
    ap.add_argument("--dice-weight", type=float, default=0.5, help="loss = CE + dice_weight * Dice")
    ap.add_argument("--class-weight-power", type=float, default=0.5,
                     help="CE class weight = (inverse-freq)**power - 1.0=raw inverse-freq (was found to make "
                          "the model spam rare-class false positives broadly), 0.5=sqrt (default, softer)")
    ap.add_argument("--max-class-weight", type=float, default=20.0,
                     help="hard cap on any single class's CE weight (None to disable) - keeps stop_line/other "
                          "from dominating the loss even after the sqrt")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True, help="체크포인트 저장 디렉토리")
    ap.add_argument("--max-steps", type=int, default=None, help="디버그/스모크 테스트용: epoch당 스텝 수 제한")
    args = ap.parse_args()

    stride = args.stride or args.patch_size

    if args.scenes:
        scene_dirs = args.scenes
    elif args.data_root:
        scene_dirs = discover_scenes(args.data_root)
    else:
        ap.error("--data-root 또는 --scenes 중 하나가 필요합니다")
    if not scene_dirs:
        ap.error("씬을 하나도 찾지 못했습니다 (mask.tif가 있는 디렉토리가 없음)")

    print(f"Loading {len(scene_dirs)} scene(s): {[d.name for d in scene_dirs]}", file=sys.stderr)
    scenes = load_scenes(scene_dirs)

    val_names = set(args.val_scenes)
    train_scene_idx = [i for i, s in enumerate(scenes) if s.name not in val_names]
    val_scene_idx = [i for i, s in enumerate(scenes) if s.name in val_names]

    if val_names and not val_scene_idx:
        ap.error(f"--val-scenes {args.val_scenes} 중 어느 것도 로드된 씬과 이름이 안 맞습니다")

    args.out.mkdir(parents=True, exist_ok=True)

    if not val_scene_idx:
        # Only one (or zero) scenes to split by - fall back to a random patch-level
        # split. Val score is optimistic here (leaks adjacent-patch context) - fine
        # for a smoke test, not for a real accuracy claim.
        print("WARNING: no --val-scenes matched (or none given) - falling back to a "
              "random patch-level train/val split within the given scene(s). This "
              "overstates validation accuracy; use --val-scenes with >=2 scenes for "
              "a real evaluation.", file=sys.stderr)
        all_patches = build_patch_index(scenes, args.patch_size, stride,
                                         args.min_valid_frac, args.bg_keep_ratio, args.seed)
        if not all_patches:
            ap.error("생성된 패치가 없습니다 (min-valid-frac을 낮춰보세요)")
        n_val = max(1, round(len(all_patches) * 0.15))
        train_patches, val_patches = all_patches[n_val:], all_patches[:n_val]
    else:
        train_scenes_set = set(train_scene_idx)
        val_scenes_set = set(val_scene_idx)
        patches = build_patch_index(scenes, args.patch_size, stride,
                                     args.min_valid_frac, args.bg_keep_ratio, args.seed)
        train_patches = [p for p in patches if p.scene_idx in train_scenes_set]
        val_patches = [p for p in patches if p.scene_idx in val_scenes_set]

    print(f"train patches: {len(train_patches)}, val patches: {len(val_patches)}", file=sys.stderr)
    if not train_patches:
        ap.error("train 패치가 0개입니다")

    train_ds = PatchDataset(scenes, train_patches, args.patch_size, augment=True)
    val_ds = PatchDataset(scenes, val_patches, args.patch_size, augment=False) if val_patches else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, drop_last=True)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
                  if val_ds else None)

    device = torch.device(args.device)
    model = UNet(in_channels=2, num_classes=NUM_CLASSES, base_channels=args.base_channels).to(device)
    weights = class_weights(scenes, NUM_CLASSES, power=args.class_weight_power,
                             max_weight=args.max_class_weight).to(device)
    print(f"class weights: {dict(zip(CLASS_NAMES.values(), weights.tolist()))}", file=sys.stderr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_miou = -1.0
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_steps = 0
        for step, (x, y) in enumerate(train_loader):
            if args.max_steps and step >= args.max_steps:
                break
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=weights, ignore_index=IGNORE_INDEX)
            if args.dice_weight:
                loss = loss + args.dice_weight * dice_loss(logits, y, NUM_CLASSES, IGNORE_INDEX)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_steps += 1
        scheduler.step()
        train_loss = running_loss / max(n_steps, 1)

        val_miou = float("nan")
        if val_loader:
            model.eval()
            conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    preds = model(x).argmax(1)
                    conf += confusion_matrix(preds.cpu(), y.cpu(), NUM_CLASSES, IGNORE_INDEX)
            ious = iou_per_class(conf)
            val_miou = float(torch.nanmean(ious))
            per_class = {name: (round(float(iou), 4) if iou == iou else None)
                         for name, iou in zip(CLASS_NAMES.values(), ious)}
            print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  "
                  f"val_mIoU={val_miou:.4f}  {per_class}  ({time.time()-t0:.1f}s)", file=sys.stderr)
        else:
            print(f"epoch {epoch+1}/{args.epochs}  loss={train_loss:.4f}  (no val set)  "
                  f"({time.time()-t0:.1f}s)", file=sys.stderr)

        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_miou": val_miou})
        torch.save(model.state_dict(), args.out / "last.pt")
        if val_miou == val_miou and val_miou > best_miou:  # NaN-safe >
            best_miou = val_miou
            torch.save(model.state_dict(), args.out / "best.pt")

    (args.out / "history.json").write_text(json.dumps(history, indent=2))
    (args.out / "config.json").write_text(json.dumps({
        "class_names": CLASS_NAMES,
        "patch_size": args.patch_size,
        "base_channels": args.base_channels,
        "in_channels": 2,
    }, indent=2))
    print(f"Done. best val mIoU={best_miou:.4f}. Checkpoints in {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
