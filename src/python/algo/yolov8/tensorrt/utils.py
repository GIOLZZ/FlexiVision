import math
import time
import numpy as np
import cv2

from algo.yolov8.results import YoloDetectResults


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

def xywh2xyxy(x):
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    """
    NMS
    Args:
        boxes (numpy.ndarray): 框坐标
        scores (numpy.ndarray): 框得分
        iou_threshold (float): iou阈值

    Returns:
        keep_boxes (numpy.ndarray): 保留的框索引
    """
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        keep_indices = np.where(ious < iou_threshold)[0]

        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    iou = intersection_area / union_area

    return iou

def extract_orig_box(box, pad, ratio, orig_shape):
    """
    将box缩放到原始图像尺寸
    Args:
        box: 模型输出box坐标
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        orig_shape: 原始图像形状
    """
    # 缩放到原始图像尺寸
    x1, y1, x2, y2 = box
    x1 = (x1 - pad[0]) / ratio
    y1 = (y1 - pad[1]) / ratio
    x2 = (x2 - pad[0]) / ratio
    y2 = (y2 - pad[1]) / ratio
    # 限制图像边界内
    x1 = max(0, min(x1, orig_shape[1]))
    y1 = max(0, min(y1, orig_shape[0]))
    x2 = max(0, min(x2, orig_shape[1]))
    y2 = max(0, min(y2, orig_shape[0]))

    orig_box = [int(x1), int(y1), int(x2), int(y2)]

    return orig_box

def postprocess_det(outputs: np.ndarray, orig_shape, conf_thres=0.25, iou_thres=0.45, ratio=1.0, pad=(0, 0)) -> YoloDetectResults:
    """
    YOLOv8Det输出后处理
    
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (height, width)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)

    Returns:
        YoloDetectResults: 结果
    """
    outputs = outputs[0]
    predictions = np.squeeze(outputs).T
    
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
        return YoloDetectResults()
    
    boxes_xyxy = xywh2xyxy(boxes)
    
    # NMS
    indices = nms(boxes_xyxy, confidences, iou_thres)
    
    # 获取结果
    orig_boxes = []
    for i in indices:
        orig_box = extract_orig_box(boxes_xyxy[i], pad, ratio, orig_shape)
        orig_boxes.append(orig_box)

    detect_results = YoloDetectResults()
    detect_results.boxes = orig_boxes
    detect_results.clss = class_ids[indices].tolist()
    detect_results.confs = confidences[indices].tolist()
    
    return detect_results

def postprocess_seg(outputs: np.ndarray, orig_shape, conf_thres=0.25, iou_thres=0.45, ratio=1.0, pad=(0, 0)) -> YoloDetectResults:
    """
    YOLOv8Seg输出后处理
    
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (height, width)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)

    Returns:
        YoloDetectResults: 结果
    """
    outputs0 = outputs[0]
    outputs1 = outputs[1]

    proto = outputs1[0] # 原型掩码，用于生成分割掩码
    nm = proto.shape[0] # 掩码系数数量
    nc = outputs0.shape[1] - 4 - 1 - nm # 类别数量
    
    predictions = np.squeeze(outputs0).T

    # conf过滤
    scores = np.max(predictions[:, 4:5+nc], axis=1)
    predictions = predictions[scores > conf_thres, :]
    confidences = scores[scores > conf_thres]

    if len(confidences) == 0:
        return YoloDetectResults()

    mask_predictions = predictions[..., nc+5:]
    box_predictions = predictions[..., :nc+5]
    boxes = box_predictions[:, :4]
    class_ids = np.argmax(box_predictions[:, 4:], axis=1)

    # 2xyxy
    boxes_xyxy = xywh2xyxy(boxes)

    # NMS
    indices = nms(boxes_xyxy, confidences, iou_thres)

    orig_boxes = []
    for i in indices:
        orig_box = extract_orig_box(boxes_xyxy[i], pad, ratio, orig_shape)
        orig_boxes.append(orig_box)

    detect_results = YoloDetectResults()
    detect_results.boxes = orig_boxes
    detect_results.clss = class_ids[indices].tolist()
    detect_results.confs = confidences[indices].tolist()

    mask_pred = mask_predictions[indices]

    # 处理速度待优化
    detect_results.masks = process_mask_output(mask_pred, outputs1, np.array(detect_results.boxes), orig_shape, (640, 640), ratio, pad)

    return detect_results

def process_mask_output(mask_predictions, mask_output, boxes, orig_shape, input_shape, ratio, pad):
    """
    Args:
        mask_predictions:
        mask_output: 模型mask输出
        boxes: boxes
        orig_shape: 原始图像形状
        input_shape: 输入图像形状
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
    Returns:
        list[np.ndarray]: 原图像尺寸mask列表
    """
    if mask_predictions.shape[0] == 0:
        return []

    mask_output = np.squeeze(mask_output)
    num_mask, mh, mw = mask_output.shape  # CHW

    # (N, num_mask) × (num_mask, mh*mw) => (N, mh*mw)
    masks = sigmoid(mask_predictions @ mask_output.reshape(num_mask, -1))
    masks = masks.reshape((-1, mh, mw))
    
    input_h, input_w = input_shape
    
    # 计算从输入图到Mask原型的缩放
    proto_r = min(mh / input_h, mw / input_w)
    
    # 计算在Mask原型空间中的完整pad
    dw, dh = pad
    pad_w_proto = dw * proto_r
    pad_h_proto = dh * proto_r
    
    # 缩放边界框并添加pad
    scale_boxes = boxes * (ratio * proto_r)
    scale_boxes[:, [0, 2]] += pad_w_proto  # x1, x2
    scale_boxes[:, [1, 3]] += pad_h_proto  # y1, y2
    
    mask_maps = np.zeros((len(scale_boxes), orig_shape[0], orig_shape[1]), dtype=np.uint8)

    for i in range(len(scale_boxes)):
        x1s, y1s, x2s, y2s = scale_boxes[i]
        x1, y1, x2, y2 = boxes[i]

        x1s, y1s = int(x1s), int(y1s)
        x2s, y2s = int(x2s), int(y2s)
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)

        # 裁剪 mask
        scale_crop_mask = masks[i][y1s:y2s, x1s:x2s]

        crop_mask = cv2.resize(
            scale_crop_mask,
            (x2 - x1, y2 - y1),
            interpolation=cv2.INTER_LINEAR
        )

        # 二值化
        crop_mask = (crop_mask > 0.5)

        mask_maps[i, y1:y2, x1:x2] = crop_mask

    return mask_maps

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


