import torch
import numpy as np
from PIL import Image
import base64
import io
import logging
import cv2
from .llm_client import LLMClient
from .config_manager import config_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageToLLMReverseNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "reverse_prompt": ("STRING", {
                    "multiline": True,
                    "default": "请描述这张图像的内容，并根据图像生成详细的提示词。",
                    "placeholder": "输入反推要求"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999999,
                    "step": 1,
                    "label": "随机种子（0表示随机）"
                }),
                "model_type": (["glm"], {
                    "default": "glm"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "输入API密钥，留空则使用已保存的密钥"
                }),
            },
            "optional": {
                "model_name": (["auto", "glm-4v", "glm-4.5v"], {
                    "default": "auto"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 1,
                    "max": 8192
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "reverse_image"
    CATEGORY = "lianlaoshi"
    OUTPUT_NODE = True

    def reverse_image(self, images, reverse_prompt, model_type, api_key, model_name="", temperature=0.7, max_tokens=2048, seed=0):
        """
        图像反推功能
        :param images: 输入图像
        :param reverse_prompt: 反推要求
        :param model_type: 模型类型
        :param api_key: API密钥
        :param model_name: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大 tokens
        :return: 反推结果
        """
        try:
            # 获取第一张图像
            image = images[0]

            # 将张量转换为PIL图像
            image = image.cpu().numpy()
            image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # 将图像转换为base64编码
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

            # 初始化LLM客户端 - 如果API密钥为空，将自动从配置中获取
            client = LLMClient(model_type=model_type, api_key=api_key)

            # 准备参数
            kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            # 如果seed大于0，则添加到参数中
            if seed > 0:
                kwargs["seed"] = seed
                logger.info(f"使用种子参数: {seed}")
            
            # 如果用户未指定模型或选择了"auto"，则自动选择最快的模型
            if model_name == "auto" or not model_name:
                # 图像反推任务包含图像输入，需要选择支持图像的最快模型
                suggested_model = client.suggest_fast_model(has_image=True)
                kwargs["model"] = suggested_model
                logger.info(f"自动选择最快模型: {suggested_model}")
            else:
                kwargs["model"] = model_name

            # 发送请求进行图像反推
            result = client.generate(
                prompt=reverse_prompt,
                image_base64=img_str,
                **kwargs
            )

            if result:
                logger.info(f"图像反推成功")
                return (result,)
            else:
                logger.error("图像反推失败，未获取到结果")
                return ("图像反推失败，未获取到结果",)

        except Exception as e:
            logger.error(f"图像反推过程中发生错误: {e}")
            return (f"图像反推过程中发生错误: {str(e)}",)

class LLMAPIKeyManager:
    """LLM API密钥管理节点"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["glm"], {"default": "glm"}),
                "api_key": ("STRING", {"default": "", "multiline": False, "placeholder": "输入API密钥"}),
                "action": (["save", "clear"], {"default": "save"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "manage_api_key"
    CATEGORY = "lianlaoshi"

    def manage_api_key(self, model_type, api_key, action):
        """
        管理LLM API密钥
        """
        try:
            if action == "save":
                if api_key:
                    success = config_manager.save_api_key("glm", api_key)
                    if success:
                        return ("成功保存GLM API密钥",)
                    else:
                        return ("保存GLM API密钥失败",)
                else:
                    return ("API密钥不能为空",)
            elif action == "clear":
                success = config_manager.clear_api_key("glm")
                if success:
                    return ("成功清除GLM API密钥",)
                else:
                    return ("清除GLM API密钥失败",)
            else:
                return ("未知操作",)
        except Exception as e:
            logger.error(f"管理API密钥时发生错误: {e}")
            return (f"管理API密钥时发生错误: {str(e)}",)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "ImageToLLMReverseNode": ImageToLLMReverseNode,
    "LLMAPIKeyManager": LLMAPIKeyManager
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageToLLMReverseNode": "lian GLM图像反推节点",
    "LLMAPIKeyManager": "lian GLM API密钥管理"
}