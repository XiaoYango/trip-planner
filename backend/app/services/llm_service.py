"""LLM服务模块"""

from hello_agents import HelloAgentsLLM
from ..config import get_settings

# 全局LLM实例
_llm_instance = None


def create_llm(timeout: int | None = None) -> HelloAgentsLLM:
    """根据统一配置创建独立的LLM客户端。"""
    settings = get_settings()
    return HelloAgentsLLM(
        model=settings.llm_model_id,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        provider=settings.llm_provider or None,
        timeout=timeout or settings.llm_timeout,
    )


def get_llm() -> HelloAgentsLLM:
    """
    获取LLM实例(单例模式)
    
    Returns:
        HelloAgentsLLM实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        # 显式传递统一配置，避免被其他OPENAI_*环境变量误导到错误的服务商。
        _llm_instance = create_llm()
        
        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")
    
    return _llm_instance


def reset_llm():
    """重置LLM实例(用于测试或重新配置)"""
    global _llm_instance
    _llm_instance = None
