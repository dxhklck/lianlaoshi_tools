import json
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PromptExpansionNode:
    def __init__(self):
        self.base_path = Path(__file__).parent
        self.config = self.load_config()

    @classmethod
    def INPUT_TYPES(cls):
        # 加载配置文件获取选项
        config_path = Path(__file__).parent / "prompt_config.json"
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            config = {}

        # 基本输入类型
        input_types = {
            "required": {
                "主体描述": ("STRING", {
                    "multiline": True, 
                    "default": "", 
                    "placeholder": f"主体描述是对主体外观特征细节的描述，可通过形容词或短句列举，"
                                   f"例如“一位身着少数民族服饰的黑发苗族少女”、"
                                   f"“一位来自异世界的飞天仙子，身着破旧却华丽的服饰，"
                                   f"背后展开一对由废墟碎片构成的奇异翅膀”。"
                                   f"主体所处环境特征细节的描述，可通过形容词或短句列举。"
                })
            },
            "optional": {}
        }

        # 从配置文件读取并添加美学控制选项
        if "美学控制" in config:
            for key, values in config["美学控制"].items():
                if isinstance(values, list) and len(values) > 0:
                    input_types["optional"][key] = ([""] + values, {"default": ""})

        # 添加动态控制选项
        if "动态控制" in config:
            for key, values in config["动态控制"].items():
                if isinstance(values, list) and len(values) > 0:
                    input_types["optional"][key] = ([""] + values, {"default": ""})

        # 添加风格化选项
        if "风格化表现" in config:
            for key, values in config["风格化表现"].items():
                if isinstance(values, list) and len(values) > 0:
                    input_types["optional"][key] = ([""] + values, {"default": ""})

        return input_types

    RETURN_TYPES = ("STRING",)
    FUNCTION = "expand_prompt"
    CATEGORY = "lianlaoshi/prompt"
    OUTPUT_NODE = True

    def load_config(self):
        try:
            config_path = self.base_path / "prompt_config.json"
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"配置文件加载失败: {e}")
            return {}

    def build_combined_prompt(self, prompt, **kwargs):
        """组合所有选项构建完整提示词"""
        components = []
        
        # 添加基础提示词
        if prompt:
            components.append(prompt)

        # 从美学控制中收集非空选项
        aesthetic_items = []
        for k, v in kwargs.items():
            if k.startswith("光源类型") and v:
                aesthetic_items.append(f"光源: {v}")
            elif k.startswith("光线类型") and v:
                aesthetic_items.append(f"光线: {v}")
            elif k.startswith("时间段") and v:
                aesthetic_items.append(f"时间: {v}")
            elif k.startswith("景别") and v:
                aesthetic_items.append(f"景别: {v}")
            elif k.startswith("构图") and v:
                aesthetic_items.append(f"构图: {v}")
            elif k.startswith("镜头焦段") and v:
                aesthetic_items.append(f"镜头: {v}")
            elif k.startswith("机位角度") and v:
                aesthetic_items.append(f"角度: {v}")
            elif k.startswith("色调") and v:
                aesthetic_items.append(f"色调: {v}")
        
        if aesthetic_items:
            components.append(", ".join(aesthetic_items))

        # 从动态控制中收集非空选项
        dynamic_items = []
        for k, v in kwargs.items():
            if k.startswith("运动类型") and v:
                dynamic_items.append(f"动作: {v}")
            elif k.startswith("人物情绪") and v:
                dynamic_items.append(f"情绪: {v}")
            elif k.startswith("基本运镜") and v:
                dynamic_items.append(f"运镜: {v}")
            elif k.startswith("高级运镜") and v:
                dynamic_items.append(f"特殊镜头: {v}")

        if dynamic_items:
            components.append(", ".join(dynamic_items))

        # 从风格化表现中收集非空选项
        style_items = []
        for k, v in kwargs.items():
            if k.startswith("视觉风格") and v:
                style_items.append(f"风格: {v}")
            elif k.startswith("特效镜头") and v:
                style_items.append(f"特效: {v}")

        if style_items:
            components.append(", ".join(style_items))

        # 组合所有组件，用逗号分隔
        final_prompt = ", ".join(filter(None, components))
        logger.info(f"组合后的提示词: {final_prompt}")
        return final_prompt

    def expand_prompt(self, 主体描述, **kwargs):
        """组合提示词"""
        try:
            # 组合所有选项的提示词
            combined_prompt = self.build_combined_prompt(主体描述, **kwargs)
            return (combined_prompt,)
        except Exception as e:
            logger.error(f"提示词组合失败: {e}")
            return (主体描述,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "PromptExpansion": PromptExpansionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptExpansion": "lian 提示词助手 微机练老师"
}