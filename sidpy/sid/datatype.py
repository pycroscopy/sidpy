from enum import Enum
class DataType(Enum):
    UNKNOWN = -1
    SPECTRUM = 1
    LINE_PLOT = 2
    LINE_PLOT_FAMILY = 3
    IMAGE = 4
    IMAGE_MAP = 5
    IMAGE_STACK = 6  # 3d
    SPECTRAL_IMAGE = 7
    IMAGE_4D = 8
    POINT_CLOUD = 9
    DP_POINT_CLOUD = 10