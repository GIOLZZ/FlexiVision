import numpy as np
# from dataclasses import dataclass


# @dataclass
class YoloDetectResults:
    """Yolo检测结果"""
    def __init__(self):
        self.boxes: list[list[int]] = []
        self.clss: list[int] = []
        self.confs: list[float] = []
        self.masks: list[np.ndarray] = []
        self.keypoints: list[list[int, float]] = []
        self.xyxyxyxy: list[list[int]] = []
