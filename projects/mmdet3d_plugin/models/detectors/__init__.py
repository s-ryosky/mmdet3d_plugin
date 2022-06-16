from .bevdet import BEVDet, BEVDetSequential
from .obj_dgcnn import ObjDGCNN
from .detr3d import Detr3D
from .petr3d import Petr3D
from .uvtr import UVTR
from .uvtr_kd_cs import UVTRKDCS
from .uvtr_kd_l import UVTRKDL
from .uvtr_kd_m import UVTRKDM

__all__ = [
    'BEVDet',
    'BEVDetSequential',
    'ObjDGCNN',
    'Detr3D',
    'Petr3D',
    'UVTR',
    'UVTRKDCS',
    'UVTRKDL',
    'UVTRKDM',
]
