"""
配置文件，包含 API 密钥和时期定义。
"""

import os
from dataclasses import dataclass
from typing import Dict

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

@dataclass(frozen=True)
class Period:
    """时期元数据。"""

    name: str
    collection_name: str
    description: str
    folder: str
    persona_file: str


# 四个时期定义
PERIODS: Dict[str, Period] = {
    "2016-2017_早期探索": Period(
        name="2016-2017 早期探索",
        collection_name="period_2016_2017",
        description="创作语汇逐步成型，尝试多种媒介与叙事。",
        folder="2016-2017_早期探索",
        persona_file="2016-2017.txt",
    ),
    "2019-2020_成熟发展": Period(
        name="2019-2020 成熟发展",
        collection_name="period_2019_2020",
        description="个人风格定型，色彩与构图更为自信。",
        folder="2019-2020_成熟发展",
        persona_file="2019-2020.txt",
    ),
    "2021_沉淀总结": Period(
        name="2021 沉淀总结",
        collection_name="period_2021",
        description="回望过往作品，梳理主题与方法，作品更收敛。",
        folder="2021_沉淀总结",
        persona_file="2021.txt",
    ),
    "2024-2025_当下状态": Period(
        name="2024-2025 当下状态",
        collection_name="period_2024_2025",
        description="与时代对话，关注现实议题与自我更新。",
        folder="2024-2025_当下状态",
        persona_file="2024-2025.txt",
    ),
}


def get_base_path() -> str:
    """返回项目根路径。"""

    return os.path.dirname(os.path.abspath(__file__))


def require_api_key() -> str:
    """确保 API 密钥存在，否则抛出友好错误。"""

    if not OPENAI_API_KEY:
        raise RuntimeError("请先在环境变量 OPENAI_API_KEY 中配置密钥。")
    return OPENAI_API_KEY
