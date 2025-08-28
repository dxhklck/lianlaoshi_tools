import logging
from .llm_client import LLMClient
from .config_manager import config_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPromptOptimizerNode:
    """通义万相视频/图片提示词优化节点"""
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
                "model_name": (["auto", "glm-4v", "glm-4.5v", "glm-4.5", "glm-4.5-flash"], {
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
                "optimization_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.1
                }),
                "optimization_type": (["video", "image"], {
                    "default": "video",
                    "label": "优化类型（视频/图片）"
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 999999999,
                    "step": 1,
                    "label": "随机种子（0表示随机）"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("optimized_prompt", "api_key")
    FUNCTION = "run"
    CATEGORY = "lianlaoshi"
    OUTPUT_NODE = True

    def run(self, input_prompt, model_type, api_key, model_name="", temperature=0.7, max_tokens=2048, optimization_strength=1.0, optimization_type="video", seed=0):
        return self.optimize_prompt(
            input_prompt=input_prompt,
            model_type=model_type,
            api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            optimization_strength=optimization_strength,
            optimization_type=optimization_type,
            seed=seed
        )
    
    def optimize_prompt(self, input_prompt, model_type, api_key, model_name="", temperature=0.7, max_tokens=2048, optimization_strength=1.0, optimization_type="video", seed=0):
        """
        根据类型（视频或图片）优化和扩写输入提示词
        :param input_prompt: 需要优化的提示词
        :param model_type: 模型类型
        :param api_key: API密钥
        :param model_name: 模型名称
        :param temperature: 温度参数
        :param max_tokens: 最大tokens
        :param optimization_strength: 优化强度 (0.5-2.0)
        :param optimization_type: 优化类型（video或image）
        :return: 优化后的提示词
        """
        try:
            # 验证输入
            if not input_prompt:
                logger.warning("输入提示词不能为空")
                return ("",)

            # 初始化LLM客户端 - 如果API密钥为空，将自动从配置中获取
            client = LLMClient(model_type=model_type, api_key=api_key)
            
            # 使用用户输入的API密钥（如果有输入）
            output_api_key = api_key

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
                # 视频提示词优化是纯文本任务，推荐使用最快的文本模型
                suggested_model = client.suggest_fast_model(has_image=False)
                kwargs["model"] = suggested_model
                logger.info(f"自动选择最快模型: {suggested_model}")
            else:
                kwargs["model"] = model_name

            # 设置扩写强度描述
            strength_desc = "适中"
            if optimization_strength < 0.8:
                strength_desc = "轻度"
            elif optimization_strength > 1.5:
                strength_desc = "高度"

            # 根据优化类型选择不同的提示词模板
            if optimization_type == "image":
                # 图片提示词模板 - 基于千问生图的特性
                prompt = f"""
                你是一个专业的图像提示词生成助手。请根据用户提供的核心概念，将其扩写成一个详细、具体、富有画面感的图像生成提示词。
                输入核心概念：{input_prompt}
                
                请严格遵循以下结构进行扩写：
                1. 主体描述：明确图像中的核心对象（人物/物体/场景）
                2. 细节刻画：详细描述外观特征、材质、颜色、神态/动作等
                3. 场景与环境：描绘主体所处背景和环境氛围
                4. 构图与视角：指定景别、视角、镜头效果
                5. 艺术风格：设定图像的艺术风格
                6. 画质与效果：描述画质和光线效果
                
                输出要求：
                - 将所有要素融合为一段连贯的描述性文字，确保逻辑流畅
                - 最终提示词应该尽可能详细，包含丰富的细节
                - 只输出最终扩写后的图像提示词，不要包含任何解释性文字、对话或前缀
                - 不要添加类似"以下是根据您提供的信息生成的图像提示词："这样的开头文本
                - 按照{strength_desc}扩写强度进行扩展，保持核心意图
                - 语言风格自然流畅，富有画面感
                - 请直接输出优化后的提示词内容，不要添加任何额外说明
                """
            else:
                # 视频提示词模板
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
                if optimization_type == "image":
                    logger.info("图像提示词优化成功")
                else:
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
                return (cleaned_prompt, output_api_key)
            else:
                if optimization_type == "image":
                    logger.error("图像提示词优化失败")
                else:
                    logger.error("通义万相视频提示词优化失败")
                return ("", output_api_key)

        except Exception as e:
            logger.error(f"通义万相视频提示词优化过程中发生错误: {e}")
            # 在异常情况下，仍然返回用户输入的API密钥
            return ("", api_key)

# 节点注册
def get_node_registrations():
    return {
        "node_class_mappings": {
            "VideoPromptOptimizerNode": VideoPromptOptimizerNode
        },
        "node_display_name_mappings": {
            "VideoPromptOptimizerNode": "lian 视频、图片提示词优化节点"
        }
    }