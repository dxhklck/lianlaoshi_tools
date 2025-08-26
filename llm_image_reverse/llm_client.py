import requests
import json
import logging
import requests
from typing import Dict, Any, Optional
from .config_manager import config_manager

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

    def generate(self, prompt: str, image_base64: str = "", **kwargs) -> Optional[str]:
        """
        向GLM模型发送请求，生成文本
        :param prompt: 提示文本
        :param image_base64: 图像的base64编码
        :param kwargs: 其他参数
        :return: 生成的文本
        """
        try:
            url = self.base_urls[self.model_type]
            payload = self._build_payload(prompt, image_base64, **kwargs)

            logger.info(f"向GLM模型发送请求...")
            response = requests.post(url, headers=self.headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            return self._parse_response(result)
        except Exception as e:
            logger.error(f"LLM请求失败: {e}")
            return None

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