# SemanticKittiDataset hardcodes its 19 KITTI class names as a class-level
# METAINFO, and Seg3DDataset.get_label_mapping() requires any custom `classes`
# passed via metainfo to be a *subset* of that hardcoded list - our classes
# (background/lane/crosswalk) aren't KITTI classes at all, so that check fails.
# Everything else about SemanticKittiDataset (the .bin/.label loading, the pkl
# info-file format) is exactly what we want; only METAINFO needs to change.
from mmdet3d.datasets import SemanticKittiDataset
from mmdet3d.registry import DATASETS


@DATASETS.register_module()
class Lane3DDataset(SemanticKittiDataset):
    METAINFO = {
        'classes': ('background', 'lane', 'crosswalk'),
        'palette': [[128, 128, 128], [220, 20, 20], [20, 120, 220]],
        'seg_valid_class_ids': (0, 1, 2),
        'seg_all_class_ids': (0, 1, 2),
    }
