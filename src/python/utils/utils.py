import cv2
import numpy as np


def expand_image_with_similar_color(img):
    """
    使用图像边缘相似颜色扩充图像
    
    Args:
        img: 输入图像
    
    Returns:
        扩充后的图像
    """
    if img is None:
        raise ValueError("无法读取图像，请检查路径是否正确")
    img_h = img.shape[0]
    top = img_h // 4
    bottom = img_h // 4
    left = img_h // 4
    right = img_h // 4
    
    # 获取原始图像尺寸
    h, w = img.shape[:2]
    
    # 创建新图像尺寸
    new_h = h + top + bottom
    new_w = w + left + right
    
    # 创建新图像
    expanded_img = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    
    # 将原图放置在新图像中心
    expanded_img[top:top+h, left:left+w] = img
    
    # 提取边缘区域用于颜色分析
    top_edge = img[0:1, :].reshape(-1, 3)
    bottom_edge = img[h-1:h, :].reshape(-1, 3)
    left_edge = img[:, 0:1].reshape(-1, 3)
    right_edge = img[:, w-1:w].reshape(-1, 3)
    
    # 计算边缘平均颜色
    top_color = np.mean(top_edge, axis=0).astype(np.uint8)
    bottom_color = np.mean(bottom_edge, axis=0).astype(np.uint8)
    left_color = np.mean(left_edge, axis=0).astype(np.uint8)
    right_color = np.mean(right_edge, axis=0).astype(np.uint8)
    
    # 填充顶部区域
    if top > 0:
        # 创建渐变效果
        for i in range(top):
            alpha = i / top
            color = (alpha * top_color + (1 - alpha) * np.mean([top_color, left_color, right_color], axis=0)).astype(np.uint8)
            expanded_img[i, left:left+w] = color
        
        # 填充左上角
        for i in range(top):
            for j in range(left):
                alpha_i = i / top
                alpha_j = j / left
                color = (alpha_i * alpha_j * top_color + 
                         alpha_i * (1 - alpha_j) * left_color + 
                         (1 - alpha_i) * alpha_j * right_color + 
                         (1 - alpha_i) * (1 - alpha_j) * np.mean([top_color, bottom_color, left_color, right_color], axis=0))
                expanded_img[i, j] = color.astype(np.uint8)
        
        # 填充右上角
        for i in range(top):
            for j in range(left+w, new_w):
                alpha_i = i / top
                alpha_j = (j - (left+w)) / right
                color = (alpha_i * alpha_j * top_color + 
                         alpha_i * (1 - alpha_j) * right_color + 
                         (1 - alpha_i) * alpha_j * left_color + 
                         (1 - alpha_i) * (1 - alpha_j) * np.mean([top_color, bottom_color, left_color, right_color], axis=0))
                expanded_img[i, j] = color.astype(np.uint8)
    
    # 填充底部区域
    if bottom > 0:
        # 创建渐变效果
        for i in range(bottom):
            alpha = i / bottom
            color = (alpha * bottom_color + (1 - alpha) * np.mean([bottom_color, left_color, right_color], axis=0)).astype(np.uint8)
            expanded_img[top+h+i, left:left+w] = color
        
        # 填充左下角
        for i in range(bottom):
            for j in range(left):
                alpha_i = i / bottom
                alpha_j = j / left
                color = (alpha_i * alpha_j * bottom_color + 
                         alpha_i * (1 - alpha_j) * left_color + 
                         (1 - alpha_i) * alpha_j * right_color + 
                         (1 - alpha_i) * (1 - alpha_j) * np.mean([top_color, bottom_color, left_color, right_color], axis=0))
                expanded_img[top+h+i, j] = color.astype(np.uint8)
        
        # 填充右下角
        for i in range(bottom):
            for j in range(left+w, new_w):
                alpha_i = i / bottom
                alpha_j = (j - (left+w)) / right
                color = (alpha_i * alpha_j * bottom_color + 
                         alpha_i * (1 - alpha_j) * right_color + 
                         (1 - alpha_i) * alpha_j * left_color + 
                         (1 - alpha_i) * (1 - alpha_j) * np.mean([top_color, bottom_color, left_color, right_color], axis=0))
                expanded_img[top+h+i, j] = color.astype(np.uint8)
    
    # 修复左侧区域填充 - 使用正确的数组形状
    if left > 0:
        for j in range(left):
            alpha = j / left
            color = (alpha * left_color + (1 - alpha) * np.mean([left_color, top_color, bottom_color], axis=0)).astype(np.uint8)
            # 创建一个形状为 (h, 3) 的数组，每一行都是color
            color_column = np.tile(color, (h, 1))
            expanded_img[top:top+h, j] = color_column
    
    # 修复右侧区域填充 - 使用正确的数组形状
    if right > 0:
        for j in range(right):
            alpha = j / right
            color = (alpha * right_color + (1 - alpha) * np.mean([right_color, top_color, bottom_color], axis=0)).astype(np.uint8)
            # 创建一个形状为 (h, 3) 的数组，每一行都是color
            color_column = np.tile(color, (h, 1))
            expanded_img[top:top+h, left+w+j] = color_column
    
    return expanded_img