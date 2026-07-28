from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agents.trip_planner_agent import MultiAgentTripPlanner
from ..models.schemas import TripPlan, TripRequest
from .state import TripWorkflowState


agent_pool = MultiAgentTripPlanner()


def normalize_request_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])

    return {
        "request": request.model_dump(),
        "status": "running",
        "validation_errors": [],
        "planner_attempts": 0,
        "feedback": "",
    }


def search_agent_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])
    query = agent_pool._build_attraction_query(request)
    result = agent_pool.attraction_agent.run(query)

    return {"attractions_raw": result}


def weather_agent_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])
    result = agent_pool.weather_agent.run(
        f"请查询{request.city}在"
        f"{request.start_date}至{request.end_date}期间的天气"
    )

    return {"weather_raw": result}


def hotel_agent_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])

    # HotelAgent 此时已经能读取 SearchAgent 的景点结果
    query = f"""
请根据下面的景点分布，推荐适合的{request.accommodation}。

城市：{request.city}
景点信息：
{state.get("attractions_raw", "")}

要求：
1. 优先靠近主要景点或公共交通
2. 返回真实名称、地址、价格区间
3. 不要虚构经纬度
"""

    result = agent_pool.hotel_agent.run(query)
    return {"hotels_raw": result}


def planner_agent_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])

    query = agent_pool._build_planner_query(
        request=request,
        attractions=state.get("attractions_raw", ""),
        weather=state.get("weather_raw", ""),
        hotels=state.get("hotels_raw", ""),
    )

    if state.get("feedback"):
        query += f"""

上一次计划需要修改，用户或校验器的反馈如下：
{state["feedback"]}
请根据反馈重新生成完整计划。
"""

    response = agent_pool.planner_agent.run(query)
    plan = agent_pool.parse_plan_strict(response, request)

    return {
        "draft": plan.model_dump(mode="json"),
        "planner_attempts": state.get("planner_attempts", 0) + 1,
        "validation_errors": [],
    }


def validate_plan_node(
    state: TripWorkflowState,
) -> dict:
    request = TripRequest.model_validate(state["request"])
    errors: list[str] = []

    try:
        plan = TripPlan.model_validate(state.get("draft"))
    except Exception as exc:
        return {
            "validation_errors": [f"计划结构不合法：{exc}"],
            "feedback": "请重新输出符合 TripPlan 数据结构的完整 JSON。",
        }

    if len(plan.days) != request.travel_days:
        errors.append(
            f"应生成 {request.travel_days} 天，实际生成 {len(plan.days)} 天"
        )

    for index, day in enumerate(plan.days):
        if not day.attractions:
            errors.append(f"第 {index + 1} 天没有景点")

        if len(day.meals) < 3:
            errors.append(f"第 {index + 1} 天缺少完整的早中晚餐")

    return {
        "validation_errors": errors,
        "feedback": "\n".join(errors),
    }


def route_after_validation(
    state: TripWorkflowState,
) -> Literal["planner", "human_review", "failed"]:
    if not state.get("validation_errors"):
        return "human_review"

    if state.get("planner_attempts", 0) < 2:
        return "planner"

    return "failed"


def human_review_node(
    state: TripWorkflowState,
) -> Command[Literal["finalize", "planner", "cancelled"]]:
    decision = interrupt({
        "type": "trip_plan_review",
        "message": "请确认旅行计划",
        "draft": state["draft"],
        "allowed_actions": [
            "approve",
            "edit",
            "replan",
            "reject",
        ],
    })

    action = decision.get("action")

    if action == "approve":
        return Command(
            update={
                "final_plan": state["draft"],
                "status": "completed",
            },
            goto="finalize",
        )

    if action == "edit":
        edited_plan = TripPlan.model_validate(
            decision["draft"]
        ).model_dump(mode="json")

        return Command(
            update={
                "draft": edited_plan,
                "final_plan": edited_plan,
                "status": "completed",
            },
            goto="finalize",
        )

    if action == "replan":
        return Command(
            update={
                "feedback": decision.get(
                    "feedback",
                    "请重新规划行程",
                ),
                "status": "running",
            },
            goto="planner",
        )

    return Command(
        update={"status": "cancelled"},
        goto="cancelled",
    )


def finalize_node(state: TripWorkflowState) -> dict:
    plan = TripPlan.model_validate(state["final_plan"])

    return {
        "final_plan": plan.model_dump(mode="json"),
        "status": "completed",
    }


def failed_node(state: TripWorkflowState) -> dict:
    return {"status": "failed"}


def cancelled_node(state: TripWorkflowState) -> dict:
    return {"status": "cancelled"}


def build_trip_graph():
    builder = StateGraph(TripWorkflowState)

    builder.add_node("normalize", normalize_request_node)
    builder.add_node("search", search_agent_node)
    builder.add_node("weather", weather_agent_node)
    builder.add_node("hotel", hotel_agent_node)
    builder.add_node("planner", planner_agent_node)
    builder.add_node("validate", validate_plan_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("failed", failed_node)
    builder.add_node("cancelled", cancelled_node)

    builder.add_edge(START, "normalize")

    # SearchAgent 与 WeatherAgent 并行执行
    builder.add_edge("normalize", "search")
    builder.add_edge("normalize", "weather")

    # 等两个并行节点都完成后，再执行 HotelAgent
    builder.add_edge(["search", "weather"], "hotel")

    builder.add_edge("hotel", "planner")
    builder.add_edge("planner", "validate")

    builder.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "planner": "planner",
            "human_review": "human_review",
            "failed": "failed",
        },
    )

    # human_review 使用 Command 动态跳转，不再添加普通出边
    builder.add_edge("finalize", END)
    builder.add_edge("failed", END)
    builder.add_edge("cancelled", END)

    return builder.compile(
        checkpointer=InMemorySaver()
    )


trip_graph = build_trip_graph()