from .centerpoint_head import CenterHeadV2
from .dgcnn3d_head import DGCNN3DHead
from .detr3d_head import Detr3DHead
from .uvtr_head import UVTRHead
from .uvtr_kd_head import UVTRKDHead

__all__ = [
    'CenterHeadV2',
    'DGCNN3DHead',
    'Detr3DHead',
    'UVTRHead',
    'UVTRKDHead',
]