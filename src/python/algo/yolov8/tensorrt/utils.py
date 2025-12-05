import numpy as np
import cv2
from algo.yolov8.results import YoloDeceteResults


def letterbox(im: np.ndarray, new_shape=(640, 640), color=(114, 114, 114), scaleFill=False, scaleup=False):
    """
    保持宽高比缩放图像并填充
    Args:
        im: 图像
        new_shape: 输出图像的形状
        color: 填充颜色
        scaleFill: 是否拉伸填充图像
        scaleup: 是否允许放大

    Returns: 处理后的图像, 缩放比例, 填充尺寸
    """
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if scaleFill:  # 拉伸填充
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        # ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    # 计算padding
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 填充
    
    return im, r, (dw, dh)

def preprocess(origin_img: np.ndarray, input_size=(640, 640)):
    """
    YOLOv8数据预处理
    Args:
        origin_img: 原图像
        input_size: 输入尺寸
    
    Returns:
        img: 输入数据
        origin_img: 原图像
        ratio: 缩放比例
        pad: 填充尺寸 (dw, dh)
    """
    if origin_img is None:
        raise FileNotFoundError(f"图像错误: {origin_img}")
    
    # Letterbox缩放
    img, ratio, pad = letterbox(origin_img, new_shape=input_size)
    
    img = img[:, :, ::-1]
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    
    return img, origin_img, ratio, pad

def postprocess_det(outputs: np.ndarray, conf_thres=0.25, iou_thres=0.45, ratio=1.0, pad=(0, 0), orig_shape=None) -> YoloDeceteResults:
    """
    YOLOv8Det输出后处理
    
    Args:
        outputs: 推理输出
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        orig_shape: 原始图像形状 (height, width)
    
    Returns:
        list: 检测结果列表
    """
    # 解析输出形状
    if len(outputs.shape) == 4:
        outputs = outputs.squeeze()
    if len(outputs.shape) == 3:
        outputs = outputs[0]

    predictions = outputs.T
    
    boxes = predictions[:, :4]  # 边界框
    scores = predictions[:, 4:]  # 类别置信度
    
    # 计算每个框的最大置信度和对应类别
    confidences = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # 置信度过滤
    mask = confidences >= conf_thres
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes_xyxy = np.zeros_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes_xyxy.tolist(),
        confidences.tolist(),
        conf_thres,
        iou_thres
    )
    
    # 获取结果
    detect_results = YoloDeceteResults()
    for i in indices:
        box = boxes_xyxy[i]
        conf = confidences[i]
        cls_id = class_ids[i]
        
        # 缩放到原始图像尺寸
        x1, y1, x2, y2 = box
        x1 = (x1 - pad[0]) / ratio
        y1 = (y1 - pad[1]) / ratio
        x2 = (x2 - pad[0]) / ratio
        y2 = (y2 - pad[1]) / ratio
        
        # 限制图像边界内
        if orig_shape is not None:
            height, width = orig_shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
        
        detect_results.boxes.append([int(x1), int(y1), int(x2), int(y2)])
        detect_results.clss.append(cls_id)
        detect_results.confs.append(conf)
    
    detect_results.masks = None
    
    return detect_results

