import torch
import numpy as np
from PIL import Image

class ImageTilingNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "rows": ("INT", {"default": 1, "min": 1, "max": 100}),
                "columns": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "tile_images"
    CATEGORY = "lianlaoshi/image"

    def tile_images(self, images, rows, columns):
        # 获取图像批次大小
        batch_size = images.shape[0]
        
        # 确保行列数不超过图像数量
        rows = min(rows, batch_size)
        columns = min(columns, batch_size)
        
        # 计算实际需要的图像数量
        total_images_needed = rows * columns
        
        # 如果图像数量不足，用黑色图像填充
        if batch_size < total_images_needed:
            padding_count = total_images_needed - batch_size
            black_image = torch.zeros_like(images[0]).unsqueeze(0)
            padding = black_image.repeat(padding_count, 1, 1, 1)
            images = torch.cat([images, padding], dim=0)
        
        # 调整图像数量以匹配网格
        images = images[:total_images_needed]
        
        # 获取单个图像的尺寸
        _, height, width, _ = images.shape
        
        # 创建拼接后的图像
        tiled_image = torch.zeros((1, height * rows, width * columns, 3), dtype=images.dtype)
        
        # 按行列拼接图像
        for i in range(rows):
            for j in range(columns):
                index = i * columns + j
                tiled_image[0, i*height:(i+1)*height, j*width:(j+1)*width, :] = images[index]
        
        return (tiled_image,)

NODE_CLASS_MAPPINGS = {
    "ImageTilingNode": ImageTilingNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTilingNode": "lian 图像按行列拼接"
}