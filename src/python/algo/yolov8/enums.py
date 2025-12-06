from enum import Enum


class DevicePlatform(Enum):
    """推理平台"""
    TORCH = 'torch'
    TENSORRT = 'tensorrt'
    ASCEND = 'ascend'
    RKNN = 'rknn'


class TorchDevice(Enum):
    """torch平台 推理设备"""
    CUDA = 'cuda'
    CPU = 'cpu'


class ModelTask(Enum):
    """模型推理任务"""
    DET = 'detect'
    SEG = 'segment'
    POSE = 'pose'
    OBB = 'obb'
    