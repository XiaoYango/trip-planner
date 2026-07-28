"""基于LangGraph的旅行规划多智能体编排。"""

import json
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agents.trip_planner_agent import get_trip_planner_agent
from ..models.schemas import TripPlan, TripRequest
from .state import TripGraphState


def _extract_trip_plan(response: str) -> TripPlan:
    """从PlannerAgent响应中严格解析TripPlan，不生成虚构的备用数据。"""
    if "```json" in response:
        json_start = response.find("```json") + 7
        json_end = response.find("```", json_start)
        if json_end == -1:
            raise ValueError("PlannerAgent返回的JSON代码块未闭合")
        json_text = response[json_start:json_end].strip()
    elif "```" in response:
        json_start = response.find("```") + 3
        json_end = response.find("```", json_start)
        if json_end == -1:
            raise ValueError("PlannerAgent返回的代码块未闭合")
        json_text = response[json_start:json_end].strip()
    elif "{" in response and "}" in response:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_text = response[json_start:json_end]
    else:
        raise ValueError("PlannerAgent响应中未找到JSON数据")

    return TripPlan.model_validate(json.loads(json_text))


def normalize_request_node(state: TripGraphState) -> dict:
    """校验并标准化旅行请求。"""
    request = TripRequest.model_validate(state["request"])
    return {
        "request": request.model_dump(),
        "status": "running",
        "attempts": 0,
        "validation_errors": [],
        "feedback": "",
    }


def search_node(state: TripGraphState) -> dict:
    """SearchAgent节点：搜索符合用户偏好的景点。"""
    request = TripRequest.model_validate(state["request"])
    planner = get_trip_planner_agent()
    preferences = "、".join(request.preferences) if request.preferences else "热门景点"
    query = (
        f"请搜索{request.city}的相关景点，用户偏好：{preferences}。"
        f"额外要求：{request.free_text_input or '无'}。"
    )

    try:
        result = planner.attraction_agent.run(query)
    except Exception as exc:
        result = f"景点搜索失败：{exc}"

    return {"attractions_raw": result}


def weather_node(state: TripGraphState) -> dict:
    """WeatherAgent节点：查询目标城市可获得的天气信息。"""
    request = TripRequest.model_validate(state["request"])
    planner = get_trip_planner_agent()
    query = (
        f"请查询{request.city}从{request.start_date}到{request.end_date}期间"
        "可获得的天气信息。如果日期超出预报范围，请明确说明无法提供准确预报。"
    )

    try:
        result = planner.weather_agent.run(query)
    except Exception as exc:
        result = f"天气查询失败：{exc}"

    return {"weather_raw": result}


def hotel_node(state: TripGraphState) -> dict:
    """HotelAgent节点：根据景点分布和住宿偏好推荐酒店。"""
    request = TripRequest.model_validate(state["request"])
    planner = get_trip_planner_agent()
    query = f"""
请为以下旅行推荐酒店：

城市：{request.city}
住宿偏好：{request.accommodation}

景点搜索结果：
{state.get("attractions_raw", "")}

优先推荐靠近主要景点集中区域的酒店。
"""

    try:
        result = planner.hotel_agent.run(query)
    except Exception as exc:
        result = f"酒店搜索失败：{exc}"

    return {"hotels_raw": result}


def planner_node(state: TripGraphState) -> dict:
    """PlannerAgent节点：整合搜索结果生成结构化旅行计划草案。"""
    request = TripRequest.model_validate(state["request"])
    planner = get_trip_planner_agent()
    query = planner._build_planner_query(
        request=request,
        attractions=state.get("attractions_raw", ""),
        weather=state.get("weather_raw", ""),
        hotels=state.get("hotels_raw", ""),
    )

    if state.get("validation_errors"):
        query += (
            "\n\n**上一次计划的校验错误，请在本次规划中修复：**\n- "
            + "\n- ".join(state["validation_errors"])
        )

    if state.get("feedback"):
        query += f"\n\n**用户反馈：**\n{state['feedback']}"

    attempts = state.get("attempts", 0) + 1

    try:
        response = planner.planner_agent.run(query)
        plan = _extract_trip_plan(response)
        return {
            "draft": plan.model_dump(mode="json"),
            "attempts": attempts,
            "validation_errors": [],
        }
    except Exception as exc:
        return {
            "draft": {},
            "attempts": attempts,
            "validation_errors": [f"PlannerAgent结果解析失败：{exc}"],
        }


def validate_plan_node(state: TripGraphState) -> dict:
    """校验草案结构及旅行天数。"""
    errors = list(state.get("validation_errors", []))

    try:
        request = TripRequest.model_validate(state["request"])
        plan = TripPlan.model_validate(state.get("draft", {}))

        if len(plan.days) != request.travel_days:
            errors.append(
                f"行程天数不一致：要求{request.travel_days}天，"
                f"实际生成{len(plan.days)}天"
            )

        for index, day in enumerate(plan.days):
            if not day.attractions:
                errors.append(f"第{index + 1}天没有景点安排")
            if not day.meals:
                errors.append(f"第{index + 1}天没有餐饮安排")
    except Exception as exc:
        if not errors:
            errors.append(f"TripPlan数据校验失败：{exc}")

    return {
        "validation_errors": errors,
        "status": "awaiting_approval" if not errors else "running",
    }


def route_after_validation(
    state: TripGraphState,
) -> Literal["planner", "human_review", "failed"]:
    """根据校验结果决定重新规划、人工确认或结束。"""
    if not state["validation_errors"]:
        return "human_review"
    if state["attempts"] < 3:
        return "planner"
    return "failed"


def human_review_node(
    state: TripGraphState,
) -> Command[Literal["finalize", "planner", "cancelled"]]:
    """暂停状态图，等待用户批准、编辑、重规划或拒绝草案。"""
    decision = interrupt(
        {
            "type": "trip_plan_review",
            "message": "请确认旅行计划草案",
            "draft": state["draft"],
            "allowed_actions": ["approve", "edit", "replan", "reject"],
        }
    )
    action = decision.get("action")

    if action == "approve":
        return Command(
            update={
                "review_decision": decision,
                "final_plan": state["draft"],
                "status": "completed",
            },
            goto="finalize",
        )

    if action == "edit":
        edited_plan = TripPlan.model_validate(decision["draft"])
        edited_draft = edited_plan.model_dump(mode="json")
        return Command(
            update={
                "review_decision": decision,
                "draft": edited_draft,
                "final_plan": edited_draft,
                "status": "completed",
            },
            goto="finalize",
        )

    if action == "replan":
        return Command(
            update={
                "review_decision": decision,
                "feedback": decision.get("feedback", ""),
                "validation_errors": [],
                "status": "running",
            },
            goto="planner",
        )

    return Command(
        update={
            "review_decision": decision,
            "status": "cancelled",
        },
        goto="cancelled",
    )


def finalize_node(state: TripGraphState) -> dict:
    """完成并再次验证最终旅行计划。"""
    plan = TripPlan.model_validate(state["final_plan"])
    return {
        "final_plan": plan.model_dump(mode="json"),
        "status": "completed",
    }


def failed_node(state: TripGraphState) -> dict:
    """标记无法生成有效计划。"""
    return {"status": "failed"}


def cancelled_node(state: TripGraphState) -> dict:
    """标记用户取消计划。"""
    return {"status": "cancelled"}


def build_trip_graph():
    """创建并编译旅行规划状态图。"""
    builder = StateGraph(TripGraphState)

    builder.add_node("normalize_request", normalize_request_node)
    builder.add_node("search", search_node)
    builder.add_node("weather", weather_node)
    builder.add_node("hotel", hotel_node)
    builder.add_node("planner", planner_node)
    builder.add_node("validate_plan", validate_plan_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("failed", failed_node)
    builder.add_node("cancelled", cancelled_node)

    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "search")
    builder.add_edge("normalize_request", "weather")
    builder.add_edge(["search", "weather"], "hotel")
    builder.add_edge("hotel", "planner")
    builder.add_edge("planner", "validate_plan")

    builder.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {
            "planner": "planner",
            "human_review": "human_review",
            "failed": "failed",
        },
    )

    builder.add_edge("finalize", END)
    builder.add_edge("failed", END)
    builder.add_edge("cancelled", END)

    return builder.compile(checkpointer=InMemorySaver())


_trip_graph = None


def get_trip_graph():
    """获取已编译的旅行规划状态图单例。"""
    global _trip_graph
    if _trip_graph is None:
        _trip_graph = build_trip_graph()
    return _trip_graph
