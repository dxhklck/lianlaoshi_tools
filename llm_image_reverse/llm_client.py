import requests
import json
import logging
from typing import Dict, Any, Optional
from .config_manager import config_manager
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_type: str = "glm", api_key: str = ""):
        """
        初始化LLM客户端
        :param model_type: 模型类型，仅支持'glm'(GLM)
        :param api_key: API密钥，如果为空则尝试从配置中获取
        """
        self.model_type = "glm"  # 强制使用GLM模型
        
        # 如果提供了API密钥，使用提供的密钥；否则从配置中获取
        self.api_key = api_key if api_key else config_manager.get_api_key(self.model_type)
        
        self.base_urls = {
            "glm": "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        }
        self.headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            self.headers["X-API-KEY"] = self.api_key
        
        # 创建可复用的会话对象，优化连接性能
        self.session = requests.Session()
        # 设置重试策略
        retry_strategy = Retry(
            total=3,  # 总重试次数
            backoff_factor=0.3,  # 重试间隔增长因子
            status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的HTTP状态码
            allowed_methods=["POST"]  # 允许重试的HTTP方法
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(self.headers)
        
        # 请求超时设置
        self.timeout = 60  # 秒
        
        # 模型性能映射（根据实际测试调整）
        self.model_performance = {
            "glm-4v": {"speed": "medium", "description": "视觉多模态，平衡性能"},
            "glm-4.5v": {"speed": "medium", "description": "视觉推理，功能更强但稍慢"},
            "glm-4.5": {"speed": "fast", "description": "旗舰模型，文本处理速度快"},
            "glm-4-9b-chat": {"speed": "fast", "description": "开源对话模型，速度快"},
            "glm-4.5-flash": {"speed": "very_fast", "description": "免费版，速度最快"}
        }
        
        # 上下文缓存（用于短时间内相同请求的快速响应）
        self.context_cache = {}
        self.cache_ttl = 30  # 缓存有效期（秒）

    def generate(self, prompt: str, image_base64: str = "", **kwargs) -> Optional[str]:
        """
        向GLM模型发送请求，生成文本
        :param prompt: 提示文本
        :param image_base64: 图像的base64编码
        :param kwargs: 其他参数
        :return: 生成的文本
        """
        try:
            start_time = time.time()
            
            # 检查是否可以使用缓存
            model_name = kwargs.get("model", "glm-4v")
            cache_key = f"{model_name}:{prompt[:100]}{'_with_image' if image_base64 else ''}"
            current_time = time.time()
            
            # 如果缓存命中且未过期，直接返回缓存结果
            if cache_key in self.context_cache and current_time - self.context_cache[cache_key]["timestamp"] < self.cache_ttl:
                cached_result = self.context_cache[cache_key]["result"]
                logger.info(f"使用缓存结果，节省了与模型的交互时间")
                return cached_result
            
            url = self.base_urls[self.model_type]
            payload = self._build_payload(prompt, image_base64, **kwargs)

            logger.info(f"向GLM模型 {model_name} 发送请求...")
            
            # 使用会话对象发送请求，设置超时
            response = self.session.post(url, data=json.dumps(payload), timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            
            parsed_result = self._parse_response(result)
            
            # 更新缓存
            if parsed_result:
                self.context_cache[cache_key] = {
                    "result": parsed_result,
                    "timestamp": current_time
                }
                # 清理过期缓存
                self._clean_cache()
            
            end_time = time.time()
            logger.info(f"模型响应时间: {end_time - start_time:.2f}秒")
            
            return parsed_result
        except requests.exceptions.Timeout:
            logger.error(f"LLM请求超时（{self.timeout}秒）")
            return "请求超时，请稍后重试"
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM请求过程中发生未知错误: {e}")
            return None
            
    def _clean_cache(self):
        """清理过期的缓存条目"""
        current_time = time.time()
        expired_keys = [
            key for key, value in self.context_cache.items()
            if current_time - value["timestamp"] >= self.cache_ttl
        ]
        for key in expired_keys:
            del self.context_cache[key]
        
    def suggest_fast_model(self, has_image: bool = False) -> str:
        """
        根据需求推荐最快的模型
        :param has_image: 是否包含图像输入
        :return: 推荐的模型名称
        """
        if has_image:
            # 有图像输入时，只能使用支持图像的模型
            # glm-4v 比 glm-4.5v 速度稍快
            return "glm-4v"
        else:
            # 纯文本任务，推荐最快的模型
            return "glm-4.5-flash"

    def _build_payload(self, prompt: str, image_base64: str, **kwargs) -> Dict[str, Any]:
        """
        构建请求 payload
        :param prompt: 提示文本
        :param image_base64: 图像的base64编码
        :param kwargs: 其他参数
        :return: 请求payload
        """
        # GLM模型的payload
        messages = [{
            "role": "user",
            "content": prompt
        }]

        if image_base64:
            messages[0]["content"] = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]

        payload = {
            "model": kwargs.get("model", "glm-4v"),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2048)
        }
        
        # 添加seed参数（如果提供）
        if "seed" in kwargs and kwargs["seed"] > 0:
            payload["seed"] = kwargs["seed"]

        return payload

    def _parse_response(self, response: Dict[str, Any]) -> Optional[str]:
        """
        解析GLM模型响应
        :param response: 模型响应
        :return: 生成的文本
        """
        if "choices" in response and len(response["choices"]) > 0:
            return response["choices"][0]["message"]["content"]

        logger.error(f"无法解析模型响应: {response}")
        return None