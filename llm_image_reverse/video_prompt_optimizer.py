import logging
from .llm_client import LLMClient
from .config_manager import config_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPromptOptimizerNode:
    """通义万相视频提示词优化节点"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入需要优化的视频提示词"
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
                "model_name": ("STRING", {
                    "default": "",
                    "placeholder": "输入模型名称，如glm-4v"
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
                "optimization_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "lianlaoshi"
    OUTPUT_NODE = True

    def optimize_prompt(self, input_prompt, model_type, api_key, model_name="", temperature=0.7, max_tokens=2048, optimization_strength=1.0):
        """
        根据通义万相视频提示词格式优化和扩写输入提示词
        :param input_prompt: 需要优化的提示词
        :param model_type: 模型类型
        :param api_key: API密钥
        :param model_name: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大tokens
        :param optimization_strength: 优化强度 (0.5-2.0)
        :return: 优化后的通义万相视频提示词
        """
        try:
            # 验证输入
            if not input_prompt:
                logger.warning("输入提示词不能为空")
                return ("",)

            # 初始化LLM客户端 - 如果API密钥为空，将自动从配置中获取
            client = LLMClient(model_type=model_type, api_key=api_key)

            # 准备参数
            kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if model_name:
                kwargs["model"] = model_name

            # 设置扩写强度描述
            strength_desc = "适中"
            if optimization_strength < 0.8:
                strength_desc = "轻度"
            elif optimization_strength > 1.5:
                strength_desc = "高度"

            # 构建优化提示 - 严格按照指定的六要素结构
            prompt = f"""
            你是一个专业的视频脚本和提示词生成助手。请根据用户提供的核心概念，将其扩写成一个详细、具体、富有画面感的视频生成提示词。
            
            输入核心概念：{input_prompt}
            
            请严格遵循以下结构进行扩写：
            1. 主体描述：详细刻画视频中的主要对象或人物的外观、特征、状态
            2. 场景描述：细致描绘主体所处环境，包括时间、地点、背景元素、光线、天气等
            3. 运动描述：明确主体的动作细节（幅度、速率、效果）
            4. 镜头语言：指定景别（如特写、近景、中景、全景）、视角（如平视、仰视、俯视）、镜头类型（如广角、长焦）、运镜方式（如推、拉、摇、移、跟、升、降）
            5. 氛围词：定义画面的情感与气氛
            6. 风格化：设定画面的艺术风格（如写实、卡通、赛博朋克、水墨画、电影感、抽象）
            
            输出要求：
            - 将所有要素融合为一段连贯的描述性文字，确保逻辑流畅
            - 最终提示词应该尽可能详细，包含丰富的细节
            - 只输出最终扩写后的视频提示词，不要包含任何解释性文字、对话或前缀
            - 不要添加类似"以下是根据您提供的信息生成的视频提示词："这样的开头文本
            - 按照{strength_desc}扩写强度进行扩展，保持核心意图
            - 语言风格自然流畅，富有画面感
            - 请直接输出优化后的提示词内容，不要添加任何额外说明
            """

            # 获取优化后的提示词
            optimized_prompt = client.generate(
                prompt=prompt,
                **kwargs
            )

            if optimized_prompt:
                logger.info("通义万相视频提示词优化成功")
                # 清理GLM模型可能返回的额外前缀文本
                cleaned_prompt = optimized_prompt.strip()
                
                # 移除常见的前缀文本
                prefixes_to_remove = [
                    "以下是根据您提供的信息生成的视频提示词：",
                    "生成的视频提示词：",
                    "视频提示词：",
                    "以下是生成的视频提示词：",
                    "优化后的视频提示词：",
                    "根据您的要求，优化后的视频提示词为：",
                    "优化后的提示词："
                ]
                
                for prefix in prefixes_to_remove:
                    if cleaned_prompt.startswith(prefix):
                        cleaned_prompt = cleaned_prompt[len(prefix):].strip()
                        break  # 移除第一个匹配的前缀后停止
                
                # 确保只返回提示词内容，去除可能的额外解释
                return (cleaned_prompt,)
            else:
                logger.error("通义万相视频提示词优化失败")
                return ("",)

        except Exception as e:
            logger.error(f"通义万相视频提示词优化过程中发生错误: {e}")
            return ("",)

# 节点注册
def get_node_registrations():
    return {
        "node_class_mappings": {
            "VideoPromptOptimizerNode": VideoPromptOptimizerNode
        },
        "node_display_name_mappings": {
            "VideoPromptOptimizerNode": "lian 视频提示词优化节点"
        }
    }