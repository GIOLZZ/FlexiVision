import numpy as np

from typing import List
from algo.yolov8.results import YoloDetectResults
from algo.yolov8.utils.utils import *


def postprocess_det(
        outputs: List[np.ndarray], 
        orig_shape, 
        conf_thres, 
        iou_thres, 
        ratio, 
        pad, 
        input_shape=(640, 640),
        keypoint_num=0
    ) -> YoloDetectResults:
    """
    YOLOv8Det输出后处理
    
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (h, )
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        input_shape (Optional): 输入形状 (h, w), seg后处理时需要
        keypoint_num (Optional): 关键点数量, pose后处理时需要

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

def postprocess_seg(
        outputs: List[np.ndarray], 
        orig_shape, 
        conf_thres, 
        iou_thres, 
        ratio, 
        pad,
        input_shape=(640, 640),
        keypoint_num=0
    ) -> YoloDetectResults:
    """
    YOLOv8-Seg输出后处理
    
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (height, width)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        input_shape (Optional): 输入形状 (h, w), seg后处理时需要
        keypoint_num (Optional): 关键点数量, pose后处理时需要

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

    # 性能待优化(CUDA加速)
    detect_results.masks = process_mask_output(mask_pred, outputs1, np.array(detect_results.boxes), orig_shape, input_shape, ratio, pad)

    return detect_results

def postprocess_obb(
        outputs: List[np.ndarray], 
        orig_shape, 
        conf_thres, 
        iou_thres, 
        ratio, 
        pad,
        input_shape=(640, 640),
        keypoint_num=0
    ) -> YoloDetectResults:
    """
    YOLOv8-Obb输出后处理
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (height, width)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        input_shape (Optional): 输入形状 (h, w), seg后处理时需要
        keypoint_num (Optional): 关键点数量, pose后处理时需要

    Returns:
        YoloDetectResults: 结果
    """
    outputs = outputs[0]
    outputs = np.squeeze(outputs)   # (5+nc+1, N)
    nc = outputs.shape[0] - 6

    boxes = outputs[:4, :].T    # (N, 4)  xywh
    angles = outputs[-1, :] # (N,)
    cls_scores = outputs[4:4+nc, :].T   # (N, nc)

    confidences = np.max(cls_scores, axis=1)
    class_ids = np.argmax(cls_scores, axis=1)
    valid_mask = confidences > conf_thres
    if valid_mask.sum() == 0:
        return YoloDetectResults()
    
    boxes = boxes[valid_mask]
    angles = angles[valid_mask]
    confidences = confidences[valid_mask]
    class_ids = class_ids[valid_mask]

    # 角度 → 度,准备cv2格式
    angles_deg = angles * 180 / np.pi
    rboxes_for_cv2 = np.hstack([boxes, angles_deg.reshape(-1, 1)])  # (M,5)
    
    # OBB NMS
    indices = nms_obb(rboxes_for_cv2, confidences, iou_thres)

    detect_results = YoloDetectResults()
    detect_results.clss = class_ids[indices].tolist()
    detect_results.confs = confidences[indices].tolist()

    rboxes = rboxes_for_cv2[indices]
    # 计算4个角点
    obb_box_list = get_obb_box(rboxes, ratio, pad, orig_shape)
    detect_results.xyxyxyxy = obb_box_list    
    
    return detect_results


def postprocess_pose(
        outputs: List[np.ndarray], 
        orig_shape, 
        conf_thres, 
        iou_thres, 
        ratio, 
        pad, 
        input_shape=(640, 640),
        keypoint_num=0
    ) -> YoloDetectResults:
    """
    YOLOv8-poes输出后处理
    Args:
        outputs: 推理输出
        orig_shape: 原始图像形状 (height, width)
        conf_thres: 置信度阈值
        iou_thres: NMS IoU阈值
        ratio: 预处理缩放比例
        pad: 填充尺寸 (dw, dh)
        input_shape (Optional): 输入形状 (h, w), seg后处理时需要
        keypoint_num (Optional): 关键点数量, pose后处理时需要

    Returns:
        YoloDetectResults: 结果
    """
    outputs = outputs[0]
    predictions = np.squeeze(outputs)   # (4+num_classes+3*keypoint_num, 8400)
    num_classes = predictions.shape[0] - 4 - 3*keypoint_num

    predictions = predictions.T
    
    # 分离不同部分
    boxes_xywh = predictions[:, 0:4]
    classes_scores = predictions[:, 4:4+num_classes]
    keypoints = predictions[:, 4+num_classes:]
    
    # 获取每个检测的最大类别分数和类别索引
    max_scores = np.max(classes_scores, axis=1)
    class_ids = np.argmax(classes_scores, axis=1)
    
    # 根据置信度阈值过滤
    keep = max_scores > conf_thres
    boxes_xywh = boxes_xywh[keep]
    max_scores = max_scores[keep]
    class_ids = class_ids[keep]
    keypoints = keypoints[keep]
    
    if len(boxes_xywh) == 0:
        return YoloDetectResults()
    
    boxes_xyxy = xywh2xyxy(boxes_xywh)
    
    # 按类别进行NMS（如果有多类别）
    unique_classes = np.unique(class_ids)
    final_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(class_ids == cls)[0]
        cls_boxes = boxes_xyxy[cls_indices]
        cls_scores = max_scores[cls_indices]
        
        if len(cls_boxes) > 0:
            nms_indices = nms(cls_boxes, cls_scores, iou_thres)
            final_indices.extend(cls_indices[nms_indices])
    
    # 没有任何框
    if len(final_indices) == 0:
        return YoloDetectResults()
    
    results = YoloDetectResults()
    for idx in final_indices:
        box = boxes_xyxy[idx]
        conf = max_scores[idx]
        cls_id = class_ids[idx]
        kpts = keypoints[idx]
        
        # 将框坐标还原到原始图像尺寸
        orig_box = extract_orig_box(box, pad, ratio, orig_shape)
        
        # 处理关键点
        orig_keypoints = []
        for i in range(0, 51, 3):
            if i >= len(kpts):
                break
                
            x = kpts[i]
            y = kpts[i+1]
            visibility = kpts[i+2]
            
            # 将关键点坐标还原到原始图像尺寸
            x = (x - pad[0]) / ratio
            y = (y - pad[1]) / ratio
            
            # 限制在图像边界内
            x = max(0, min(x, orig_shape[1]))
            y = max(0, min(y, orig_shape[0]))
            
            # 根据可见性阈值过滤
            if visibility > 0.5:
                orig_keypoints.append([int(x), int(y), float(visibility)])
            else:
                orig_keypoints.append([0.0, 0.0, 0.0])
        
        results.boxes.append(orig_box)
        results.clss.append(int(cls_id))
        results.confs.append(float(conf))
        results.keypoints.append(orig_keypoints)
    
    return results
