# Cylinder3D adapted for 3-class ground-point segmentation (background/lane/
# crosswalk) on tiled point clouds cut from an aggregated MMS corridor map, with
# ground truth from HD map SHP layers (see ../prepare_dataset.py).
#
# Differs from the stock SemanticKITTI config in ways that matter:
#   - 3 classes instead of 19, class_weight upweighting the rare lane/crosswalk
#     classes (each tile is dominated by plain background ground points)
#   - point_cloud_range/grid_shape shrunk to match our ~30m-wide tiles, vs. the
#     original's 50m-radius single-sensor-scan assumption
#   - no LaserMix/PolarMix (those mix between two full scans; skipped for this
#     first end-to-end validation run - the base flip/rotate/scale augmentation
#     from the stock config is kept)
_base_ = [
    '/home/ysc/miniconda3/envs/lane3d/lib/python3.10/site-packages/mmdet3d/.mim/configs/_base_/default_runtime.py',
]
# registers Lane3DDataset (see ../lane3d_dataset.py) - SemanticKittiDataset's
# METAINFO hardcodes the 19 stock KITTI class names, so our classes need a
# subclass with its own METAINFO rather than reusing SemanticKittiDataset as-is
custom_imports = dict(imports=['lane3d_dataset'], allow_failed_imports=False)

class_names = ('background', 'lane', 'crosswalk')
num_classes = len(class_names)

grid_shape = [160, 120, 24]
point_cloud_range = [0, -3.14159265359, -4, 25, 3.14159265359, 4]

model = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=point_cloud_range,
            max_num_points=-1,
            max_voxels=-1,
        ),
    ),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        return_point_feats=False),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='Cylinder3DHead',
        channels=128,
        num_classes=num_classes,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            # background/lane/crosswalk are 93.1/4.6/2.3% of points across the
            # full 14-file dataset - upweight the rare classes so the loss
            # doesn't just learn to predict background everywhere
            class_weight=[1.0, 6.0, 8.0],
            loss_weight=1.0),
        loss_lovasz=dict(type='LovaszLoss', loss_weight=1.0, reduction='none'),
    ),
    train_cfg=None,
    test_cfg=dict(mode='whole'),
)

# ---- dataset ----
dataset_type = 'Lane3DDataset'
data_root = '/home/ysc/potree/training/data/lane3d'

metainfo = dict(
    classes=class_names,
    palette=[[128, 128, 128], [220, 20, 20], [20, 120, 220]],
    seg_label_mapping={0: 0, 1: 1, 2: 2},
    max_label=2,
)

input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4, backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='lane3d_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='lane3d_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# ---- schedule (real run on the full 14-file / 2369-tile dataset) ----
lr = 0.001
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=3)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', begin=0, end=30, by_epoch=True, milestones=[20, 26], gamma=0.1),
]

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=3, max_keep_ckpts=5))
work_dir = '/home/ysc/potree/training/work_dirs/cylinder3d_lane3d_full'
