import numpy as np

from ultralytics import YOLO

from algo.yolov8.enums import ModelTask, TorchDevice
from algo.yolov8.results import YoloDetectResults
from algo.yolov8.torch.predict import DetectionPredictor, SegmentationPredictor


class Yolov8Torch:
    """YoloV8 Torch 原库推理"""
    def __init__(self, model_path: str, task: ModelTask, device: TorchDevice=TorchDevice.CPU):
        """
        Args:
            model_path (str): path
            task (ModelTask): 模型任务. 目前只支持'detect'、'segment'
            device (TorchDevice, optional): 推理设备. Defaults to TorchDevice.CPU.
        """
        self.model = YOLO(model_path)
        self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False, device=device.value)
        self.task = task

        if not isinstance(self.task, ModelTask):
            raise ValueError("模型任务错误")
        
        if self.task == ModelTask.DET:
            self.predictor = DetectionPredictor(self.model, device.value)
        elif self.task == ModelTask.SEG:
            self.predictor = SegmentationPredictor(self.model, device.value)

        print('CUDA加载模型成功:', self.model.info())

    def detect(self, image: np.ndarray, conf: float=0.25, iou: float=0.45) -> YoloDetectResults:
        results = self.predictor.predict(image, conf, iou)

        return results











