# config.py
from dataclasses import dataclass

# ===== DeepSeek 配置 =====
DEEPSEEK_API_KEY = "xxxxxx"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


@dataclass
class Settings:
    # 理赔业务阈值
    approval_amount_threshold: float = 300.0  # 超过这个金额建议人工审核
    high_risk_level: str = "high"            # high 风险一定要人工审核

    # Streaming 相关（Graph 自身 streaming）
    enable_streaming: bool = True

settings = Settings()
