import os
import json
import logging
from pathlib import Path
import dotenv

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.config_dir = self.base_path / "config"
        self.env_file = self.config_dir / ".env"
        self.api_keys_file = self.config_dir / "api_keys.json"
        
        # 确保配置目录存在
        self._ensure_config_dir()
        
        # 加载环境变量
        self._load_env()
    
    def _ensure_config_dir(self):
        """确保配置目录存在"""
        if not self.config_dir.exists():
            try:
                self.config_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"创建配置目录: {self.config_dir}")
            except Exception as e:
                logger.error(f"创建配置目录失败: {e}")
    
    def _load_env(self):
        """加载环境变量"""
        try:
            if self.env_file.exists():
                dotenv.load_dotenv(self.env_file)
                logger.info(f"加载环境变量成功: {self.env_file}")
        except Exception as e:
            logger.error(f"加载环境变量失败: {e}")
    
    def save_api_key(self, api_key_type: str, api_key: str) -> bool:
        """
        保存API密钥
        :param api_key_type: API密钥类型，如'qwen'或'glm'
        :param api_key: API密钥
        :return: 是否保存成功
        """
        try:
            # 读取现有配置
            config = {}
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 只有当文件内容非空时才尝试解析JSON
                        config = json.loads(content)
            
            # 更新API密钥
            config[api_key_type] = api_key
            
            # 保存配置
            with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            # 设置环境变量
            env_key = f"LLM_{api_key_type.upper()}_API_KEY"
            os.environ[env_key] = api_key
            
            logger.info(f"保存{api_key_type} API密钥成功")
            return True
        except Exception as e:
            logger.error(f"保存{api_key_type} API密钥失败: {e}")
            return False
    
    def get_api_key(self, api_key_type: str) -> str:
        """
        获取API密钥
        :param api_key_type: API密钥类型，如'qwen'或'glm'
        :return: API密钥，如果未找到则返回空字符串
        """
        try:
            # 首先尝试从环境变量获取
            env_key = f"LLM_{api_key_type.upper()}_API_KEY"
            if env_key in os.environ:
                return os.environ[env_key]
            
            # 然后尝试从配置文件获取
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    if api_key_type in config:
                        # 设置环境变量以便后续使用
                        os.environ[env_key] = config[api_key_type]
                        return config[api_key_type]
            
            logger.info(f"未找到{api_key_type} API密钥")
            return ""
        except Exception as e:
            logger.error(f"获取{api_key_type} API密钥失败: {e}")
            return ""
    
    def clear_api_key(self, api_key_type: str) -> bool:
        """
        清除API密钥
        :param api_key_type: API密钥类型，如'qwen'或'glm'
        :return: 是否清除成功
        """
        try:
            # 从环境变量中删除
            env_key = f"LLM_{api_key_type.upper()}_API_KEY"
            if env_key in os.environ:
                del os.environ[env_key]
            
            # 从配置文件中删除
            if self.api_keys_file.exists():
                with open(self.api_keys_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                if api_key_type in config:
                    del config[api_key_type]
                    with open(self.api_keys_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"清除{api_key_type} API密钥成功")
            return True
        except Exception as e:
            logger.error(f"清除{api_key_type} API密钥失败: {e}")
            return False

# 创建全局配置管理器实例
config_manager = ConfigManager()