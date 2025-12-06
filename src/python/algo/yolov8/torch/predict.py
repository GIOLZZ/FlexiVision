import numpy as np
from ultralytics import YOLO

from algo.yolov8.results import YoloDetectResults
from utils.utils import scale_image


class DetectionPredictor:
    def __init__(self, model: YOLO, device: str):
        self.model = model
        self.device = device

    def predict(self, image: np.ndarray, conf: float=0.25, iou: float=0.1) -> YoloDetectResults:
        """
        Args:
            image (np.ndarray): 输入图像
            conf (float, optional): 置信度. Defaults to 0.25.
            iou (float, optional): iou. Defaults to 0.1.
        
        Returns:
            YoloDetectResults: 检测结果
        """
        detect_results = YoloDetectResults()

        results = self.model(image, conf=conf, iou=iou, verbose=False, device=self.device)
        detect_results.boxes = results[0].boxes.xyxy.int().tolist()
        detect_results.clss = results[0].boxes.cls.int().tolist()
        detect_results.confs = results[0].boxes.conf.tolist()

        return detect_results


class SegmentationPredictor:
    def __init__(self, model: YOLO, device: str):
        self.model = model
        self.device = device

    def predict(self, image: np.ndarray, conf: float=0.25, iou: float=0.1) -> YoloDetectResults:
        """
        Args:
            image (np.ndarray): 输入图像
            conf (float, optional): 置信度. Defaults to 0.25.
            iou (float, optional): iou. Defaults to 0.1.
        
        Returns:
            YoloDetectResults: 检测结果
        """
        detect_results = YoloDetectResults()

        results = self.model(image, conf=conf, iou=iou, verbose=False, device=self.device)
        if results[0].masks is not None:
            results_mask_data = results[0].masks.data
            detect_results.boxes = results[0].boxes.xyxy.int().tolist()
            detect_results.clss = results[0].boxes.cls.int().tolist()
            detect_results.confs = results[0].boxes.conf.tolist()
            # 处理results_mask_data
            for mask in results_mask_data:
                mask = mask.cpu().numpy().astype(np.uint8)
                mask_resized = scale_image(mask, image.shape)  # 将mask缩放至img相同大小，此处不能简单使用cv2.resize()
                detect_results.masks.append(mask_resized)
    
        return detect_results
