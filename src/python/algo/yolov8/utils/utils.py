import math
import numpy as np
import cv2


def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size.

    Args:
        masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
        im0_shape (tuple): the original image shape
        ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
        masks (torch.Tensor): The masks that are being returned.
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        # gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

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

def process_mask_output(mask_predictions, mask_output, boxes, orig_shape, input_shape, ratio, pad):
    """
    处理Yolov8-seg mask输出
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

def obb_to_polygon(box):
    """[cx, cy, w, h, angle] to 4x2 polygon"""
    cx, cy, w, h, angle = box
    rect = ((cx, cy), (w, h), angle)
    points = cv2.boxPoints(rect)  # 4x2
    return points

def polygon_iou(poly1, poly2):
    """Compute IoU of two polygons using OpenCV"""
    poly1 = poly1.astype(np.float32)
    poly2 = poly2.astype(np.float32)

    inter = cv2.intersectConvexConvex(poly1, poly2)

    if inter[0] <= 0:
        return 0.0

    inter_area = inter[0]
    area1 = cv2.contourArea(poly1)
    area2 = cv2.contourArea(poly2)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area

def compute_iou_obb(box, boxes):
    """计算OBB的IOU"""
    poly1 = obb_to_polygon(box)
    ious = np.zeros(len(boxes), dtype=np.float32)

    for i, b in enumerate(boxes):
        poly2 = obb_to_polygon(b)
        ious[i] = polygon_iou(poly1, poly2)

    return ious

def nms_obb(boxes, scores, iou_threshold):
    """
    NMS OBB
    Args:
        boxes (numpy.ndarray): 框坐标
        scores (numpy.ndarray): 框得分
        iou_threshold (float): iou阈值

    Returns:
        keep_boxes (numpy.ndarray): 保留的框索引
    """
    order = scores.argsort()[::-1]  # descending
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        if len(order) == 1:
            break

        ious = compute_iou_obb(boxes[i], boxes[order[1:]])

        remain = np.where(ious < iou_threshold)[0]
        order = order[remain + 1]

    return keep

def get_obb_box(rboxes, ratio, pad, original_size):
    """
    将 YOLOv8 旋转框从 letterbox 坐标恢复到原图坐标

    Args:
        rboxes: (N, 5) 预测框 (x, y, w, h, angleDeg)
        ratio: letterbox 的缩放比例 r
        pad: letterbox 的 padding (dw, dh)
        original_size: (H, W) 原图大小

    Returns:
        box_list: 每个框的 4 个角点 [[x1,y1], ...]
    """
    dw, dh = pad                # letterbox padding
    H0, W0 = original_size      # 原图尺寸
    box_list = []

    for box in rboxes:
        x, y, w, h, angle_deg = box
        r = np.deg2rad(angle_deg)
        corners = rotate_box(x, y, w, h, r)

        corners[:, 0] -= dw
        corners[:, 1] -= dh

        corners /= ratio

        # 限制边界
        corners[:, 0] = np.clip(corners[:, 0], 0, W0 - 1)
        corners[:, 1] = np.clip(corners[:, 1], 0, H0 - 1)

        box_list.append(corners.astype(int).tolist())

    return box_list

def rotate_box(x, y, w, h, r):
    """
    计算旋转框的四个角点坐标。
    Args:
        x, y: 旋转框的中心坐标
        w, h: 旋转框的宽度和高度
        r: 旋转角度（弧度）
    Returns:
        四个角点的坐标，按照顺时针顺序排列
    """
    # 计算框的四个角点
    cos_r = math.cos(r)
    sin_r = math.sin(r)
    
    # 相对于框中心的四个角点坐标
    corners = np.array([
        [-w / 2, -h / 2],  # 左上角
        [w / 2, -h / 2],   # 右上角
        [w / 2, h / 2],    # 右下角
        [-w / 2, h / 2],   # 左下角
    ])

    # 旋转矩阵
    rotation_matrix = np.array([
        [cos_r, -sin_r],
        [sin_r, cos_r]
    ])
    
    # 旋转并平移到(x, y)
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x, y])
    
    return rotated_corners
