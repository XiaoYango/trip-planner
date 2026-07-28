from typing import Any, Literal
from typing_extensions import TypedDict, NotRequired


class TripGraphState(TypedDict):
    # 输入
    request: dict[str, Any]

    # Agent 中间结果
    attractions_raw: NotRequired[str]
    weather_raw: NotRequired[str]
    hotels_raw: NotRequired[str]

    # Planner 结果
    draft: NotRequired[dict[str, Any]]
    final_plan: NotRequired[dict[str, Any]]

    # 流程控制
    status: Literal[
        "running",
        "awaiting_approval",
        "completed",
        "cancelled",
        "failed",
    ]
    attempts: int
    validation_errors: list[str]
    feedback: str

    # 人工决定
    review_decision: NotRequired[dict[str, Any]]