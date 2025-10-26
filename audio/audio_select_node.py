import torch
from typing import Dict, Any, Tuple

class AudioSelectNode:  

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "audio_list": ("AUDIO_LIST",),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "number",
                }),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "select"
    CATEGORY = "lianlaoshi/audio"

    def select(self, audio_list, index: int) -> Tuple[Dict[str, Any],]:
        # 空列表兜底处理
        if not isinstance(audio_list, list) or len(audio_list) == 0:
            return ({"waveform": torch.empty(0, 0, 0), "sample_rate": 0},)

        # 安全夹取索引到有效范围 [0, len-1]
        idx = max(0, min(int(index), len(audio_list) - 1))
        selected = audio_list[idx]

        # 直接返回选中的段（保持原有 AUDIO 字典结构）
        return (selected,)


# 节点注册
NODE_CLASS_MAPPINGS = {
    "AudioSelectNode": AudioSelectNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSelectNode": "lian 音频选择节点",
}