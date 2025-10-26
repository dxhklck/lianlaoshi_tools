import numpy as np

class LianVideoResolutionNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入宽度": ("INT", {"default": 720, "min": 1, "max": 8192}),
                "输入高度": ("INT", {"default": 1280, "min": 1, "max": 8192}),
                "最低分辨率": ("INT", {"default": 720, "min": 1, "max": 8192}),
            },
        }

    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("输出宽度", "输出高度",)
    FUNCTION = "process"
    CATEGORY = "lianlaoshi/video"

    def process(self, 输入宽度, 输入高度, 最低分辨率):
        # 判断是否为竖屏
        is_portrait = 输入高度 > 输入宽度
        
        # 计算目标宽高比 (9:16)
        target_ratio = 9/16
        
        # 根据最低分辨率和方向计算新的宽高
        if is_portrait:
            # 竖屏：高度是较大值
            new_height = max(最低分辨率, int(最低分辨率 * (16/9)))
            new_width = int(new_height * (9/16))
        else:
            # 横屏：宽度是较大值
            new_width = max(最低分辨率, int(最低分辨率 * (16/9)))
            new_height = int(new_width * (9/16))
            
        return (new_width, new_height,)

# 这个方法用于注册节点
NODE_CLASS_MAPPINGS = {
    "LianVideoResolution": LianVideoResolutionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LianVideoResolution": "lian 视频横竖屏宽高自动处理"
}

