import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np

from algo.yolov8.results import YoloDetectResults
from algo.yolov8.enums import DevicePlatform, ModelTask, TorchDevice



class Yolov8Decete:
    """Yolov8 全任务检测"""
    def __init__(self, model_path: str, task: ModelTask, platform: DevicePlatform=DevicePlatform.TORCH, torch_device: TorchDevice=TorchDevice.CPU):
        """
        Args:
            model_path (str): path
            task (ModelTask): 模型任务
            platform (DevicePlatform, optional): 推理平台. Defaults to DevicePlatform.TORCH.
            torch_device (TorchDevice, optional): torch 推理设备. Defaults to TorchDevice.CPU.
        """
        if platform == DevicePlatform.TORCH:
            from algo.yolov8.torch.yolov8_torch import Yolov8Torch
            self.yolov8_model = Yolov8Torch(model_path, task=task, device=torch_device)

        elif platform == DevicePlatform.TENSORRT:
            from algo.yolov8.tensorrt.yolov8_trt import Yolov8Trt
            self.yolov8_model = Yolov8Trt(model_path, task=task)
        
        elif platform == DevicePlatform.ASCEND:
            pass

        elif platform == DevicePlatform.RKNN:
            pass

    def detect(self, image: np.ndarray, conf: float=0.25, iou: float=0.45) -> YoloDetectResults:
        """
        Args:
            image (np.ndarray): 输入图像
            conf (float, optional): 置信度. Defaults to 0.25.
            iou (float, optional): iou. Defaults to 0.45.
        
        Returns:
            YoloDetectResults: 检测结果
        """
        results = self.yolov8_model.detect(image, conf=conf, iou=iou)

        return results


if __name__ == "__main__":
    import cv2
    import time

    yolov8_detect = Yolov8Decete(
        model_path="/home/cc/FlexiVision/models/v8sseg_gaosu_0922.engine",
        task=ModelTask.SEG,
        platform=DevicePlatform.TENSORRT
    )

    image = cv2.imread("/home/cc/FlexiVision/000270.jpg")
    for _ in range(1):
        t = time.time()
        results = yolov8_detect.detect(image, conf=0.25, iou=0.8)
        print(time.time() - t)

    for box in results.boxes:
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # for mask in results.masks:
    #     mask[mask > 0] = 255
    #     cv2.imshow("mask", mask)
    #     cv2.waitKey(0)
    #     cv2.imwrite("/home/cc/FlexiVision/mask.png", mask)
        

    img_show = cv2.resize(image, (720, 480))
    cv2.imshow("result", img_show)
    cv2.waitKey(0)