"""旅行规划状态图的共享状态定义。"""

from typing import Any, Literal
from typing_extensions import NotRequired, TypedDict


WorkflowStatus = Literal[
    "running",
    "awaiting_approval",
    "completed",
    "cancelled",
    "failed",
]


class TripGraphState(TypedDict):
    """LangGraph各节点共享的旅行规划状态。"""

    request: dict[str, Any]
    status: WorkflowStatus
    attempts: int
    validation_errors: list[str]
    feedback: str
    generation_mode: NotRequired[Literal["standard", "daily"]]

    attractions_raw: NotRequired[str]
    weather_raw: NotRequired[str]
    hotels_raw: NotRequired[str]
    daily_plans: NotRequired[list[dict[str, Any]]]

    draft: NotRequired[dict[str, Any]]
    final_plan: NotRequired[dict[str, Any]]
    review_decision: NotRequired[dict[str, Any]]
