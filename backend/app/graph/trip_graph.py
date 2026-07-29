"""基于LangGraph的旅行规划多智能体编排。"""

import json
from datetime import date, timedelta
from typing import Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..agents.trip_planner_agent import get_trip_planner_agent
from ..models.schemas import Budget, DayPlan, TripPlan, TripRequest, WeatherInfo
from .state import TripGraphState


def _emit_stream_event(event: str, **data) -> None:
    """仅在LangGraph流式执行时发送自定义事件。"""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"event": event, **data})


def _extract_json_object(response: str) -> dict:
    """从模型回复中提取一个JSON对象。"""
    if "```json" in response:
        json_start = response.find("```json") + 7
        json_end = response.find("```", json_start)
        if json_end == -1:
            raise ValueError("JSON代码块未闭合")
        json_text = response[json_start:json_end].strip()
    elif "```" in response:
        json_start = response.find("```") + 3
        json_end = response.find("```", json_start)
        if json_end == -1:
            raise ValueError("代码块未闭合")
        json_text = response[json_start:json_end].strip()
    elif "{" in response and "}" in response:
        json_start = response.find("{")
        json_end = response.rfind("}") + 1
        json_text = response[json_start:json_end]
    else:
        raise ValueError("模型响应中未找到JSON数据")

    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("模型返回的JSON不是对象")
    return payload


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
    _emit_stream_event(
        "status",
        node="search",
        message="正在搜索符合偏好的景点...",
        progress=10,
    )
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

    _emit_stream_event(
        "status",
        node="search",
        message="景点信息搜索完成",
        progress=30,
    )
    return {"attractions_raw": result}


def weather_node(state: TripGraphState) -> dict:
    """WeatherAgent节点：查询目标城市可获得的天气信息。"""
    _emit_stream_event(
        "status",
        node="weather",
        message="正在查询旅行期间天气...",
        progress=10,
    )
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

    _emit_stream_event(
        "status",
        node="weather",
        message="天气信息查询完成",
        progress=30,
    )
    return {"weather_raw": result}


def hotel_node(state: TripGraphState) -> dict:
    """HotelAgent节点：根据景点分布和住宿偏好推荐酒店。"""
    _emit_stream_event(
        "status",
        node="hotel",
        message="正在根据景点分布推荐酒店...",
        progress=40,
    )
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

    _emit_stream_event(
        "status",
        node="hotel",
        message="酒店信息准备完成",
        progress=50,
    )
    return {"hotels_raw": result}


def _build_daily_query(
    state: TripGraphState,
    request: TripRequest,
    day_index: int,
    day_date: str,
    completed_days: list[DayPlan],
) -> str:
    """构建单日结构化行程生成请求。"""
    completed_summary = "\n".join(
        f"- 第{day.day_index + 1}天："
        + "、".join(attraction.name for attraction in day.attractions)
        for day in completed_days
    ) or "无"
    preferences = "、".join(request.preferences) if request.preferences else "热门景点"

    return f"""
你正在为用户逐日生成旅行计划。现在只生成第{day_index + 1}天，不要生成其他日期。

用户需求：
- 城市：{request.city}
- 日期：{day_date}
- 总旅行天数：{request.travel_days}
- 交通方式：{request.transportation}
- 住宿偏好：{request.accommodation}
- 旅行偏好：{preferences}
- 额外要求：{request.free_text_input or "无"}

已经生成的日期与景点，当前日期应尽量避免重复：
{completed_summary}

景点搜索资料：
{state.get("attractions_raw", "")}

天气资料：
{state.get("weather_raw", "")}

酒店资料：
{state.get("hotels_raw", "")}

仅返回一个JSON对象，不要使用Markdown。格式必须严格如下：
{{
  "day_plan": {{
    "date": "{day_date}",
    "day_index": {day_index},
    "description": "当日路线概述",
    "transportation": "{request.transportation}",
    "accommodation": "{request.accommodation}",
    "hotel": {{
      "name": "酒店名称",
      "address": "地址",
      "location": {{"longitude": 116.0, "latitude": 39.0}},
      "price_range": "价格范围",
      "rating": "评分",
      "distance": "距离",
      "type": "{request.accommodation}",
      "estimated_cost": 0
    }},
    "attractions": [
      {{
        "name": "景点名称",
        "address": "景点地址",
        "location": {{"longitude": 116.0, "latitude": 39.0}},
        "visit_duration": 120,
        "description": "游览说明",
        "category": "景点类别",
        "rating": 0,
        "photos": [],
        "poi_id": "",
        "image_url": null,
        "ticket_price": 0
      }}
    ],
    "meals": [
      {{
        "type": "lunch",
        "name": "餐饮名称",
        "address": "地址",
        "location": null,
        "description": "餐饮说明",
        "estimated_cost": 0
      }}
    ]
  }},
  "weather": {{
    "date": "{day_date}",
    "day_weather": "未知或天气资料中的白天天气",
    "night_weather": "未知或天气资料中的夜间天气",
    "day_temp": 0,
    "night_temp": 0,
    "wind_direction": "",
    "wind_power": ""
  }}
}}

要求：
1. 每天安排2到3个相互距离合理的景点。
2. 必须包含餐饮信息。
3. 只能依据提供的工具资料；资料不足时明确保守处理，不编造评分或价格。
4. 经纬度必须是数字。
"""


def _calculate_budget(days: list[DayPlan]) -> Budget:
    """根据逐日结构化数据汇总可计算的预算。"""
    attractions = sum(
        attraction.ticket_price
        for day in days
        for attraction in day.attractions
    )
    meals = sum(
        meal.estimated_cost
        for day in days
        for meal in day.meals
    )
    hotels = sum(
        day.hotel.estimated_cost
        for day in days[:-1]
        if day.hotel is not None
    )
    transportation = 0
    return Budget(
        total_attractions=attractions,
        total_hotels=hotels,
        total_meals=meals,
        total_transportation=transportation,
        total=attractions + hotels + meals + transportation,
    )


def daily_planner_node(state: TripGraphState) -> dict:
    """逐日生成行程，并在每一天完成后立即发送流式事件。"""
    request = TripRequest.model_validate(state["request"])
    planner = get_trip_planner_agent()
    start = date.fromisoformat(request.start_date)
    completed_days: list[DayPlan] = []
    weather_info: list[WeatherInfo] = []
    attempts = state.get("attempts", 0) + 1

    _emit_stream_event(
        "status",
        node="daily_planner",
        message="开始逐日生成行程...",
        progress=55,
        total_days=request.travel_days,
    )

    for day_index in range(request.travel_days):
        day_date = (start + timedelta(days=day_index)).isoformat()
        _emit_stream_event(
            "status",
            node="daily_planner",
            message=f"正在生成第{day_index + 1}天行程...",
            progress=55 + int(day_index / request.travel_days * 35),
            day=day_index + 1,
            total_days=request.travel_days,
        )

        last_error: Exception | None = None
        for daily_attempt in range(1, 3):
            try:
                query = _build_daily_query(
                    state,
                    request,
                    day_index,
                    day_date,
                    completed_days,
                )
                response = planner.planner_agent.run(query)
                payload = _extract_json_object(response)
                day_plan = DayPlan.model_validate(payload["day_plan"]).model_copy(
                    update={
                        "date": day_date,
                        "day_index": day_index,
                        "transportation": request.transportation,
                        "accommodation": request.accommodation,
                    }
                )
                weather_payload = payload.get("weather")
                if weather_payload:
                    weather = WeatherInfo.model_validate(weather_payload).model_copy(
                        update={"date": day_date}
                    )
                    weather_info.append(weather)

                completed_days.append(day_plan)
                _emit_stream_event(
                    "day",
                    day=day_index + 1,
                    total_days=request.travel_days,
                    plan=day_plan.model_dump(mode="json"),
                    progress=55 + int((day_index + 1) / request.travel_days * 35),
                )
                break
            except Exception as exc:
                last_error = exc
                if daily_attempt < 2:
                    _emit_stream_event(
                        "status",
                        node="daily_planner",
                        message=f"第{day_index + 1}天生成失败，正在重试...",
                        day=day_index + 1,
                        total_days=request.travel_days,
                    )
        else:
            error = f"第{day_index + 1}天生成失败：{last_error}"
            _emit_stream_event("warning", message=error, day=day_index + 1)
            return {
                "draft": {},
                "daily_plans": [
                    day.model_dump(mode="json") for day in completed_days
                ],
                "attempts": attempts,
                "validation_errors": [error],
            }

    suggestions = (
        f"本行程按{request.transportation}设计，请根据实时天气、开放时间和交通情况"
        "适当调整。热门景点建议提前预约。"
    )
    plan = TripPlan(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        days=completed_days,
        weather_info=weather_info,
        overall_suggestions=suggestions,
        budget=_calculate_budget(completed_days),
    )
    _emit_stream_event(
        "status",
        node="validate_plan",
        message="每日行程已生成，正在汇总校验...",
        progress=95,
    )
    return {
        "draft": plan.model_dump(mode="json"),
        "daily_plans": [
            day.model_dump(mode="json") for day in completed_days
        ],
        "attempts": attempts,
        "validation_errors": [],
    }


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


def route_after_hotel(
    state: TripGraphState,
) -> Literal["planner", "daily_planner"]:
    """流式请求走逐日规划，旧接口继续走完整Planner。"""
    if state.get("generation_mode") == "daily":
        return "daily_planner"
    return "planner"


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
) -> Literal["planner", "daily_planner", "human_review", "failed"]:
    """根据校验结果决定重新规划、人工确认或结束。"""
    if not state["validation_errors"]:
        return "human_review"
    if state["attempts"] < 3:
        if state.get("generation_mode") == "daily":
            return "daily_planner"
        return "planner"
    return "failed"


def human_review_node(
    state: TripGraphState,
) -> Command[Literal["finalize", "planner", "daily_planner", "cancelled"]]:
    """暂停状态图，等待用户批准、编辑、重规划或拒绝草案。"""
    _emit_stream_event(
        "status",
        node="human_review",
        message="完整计划校验通过，等待人工确认",
        progress=100,
    )
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
        replan_target = (
            "daily_planner"
            if state.get("generation_mode") == "daily"
            else "planner"
        )
        return Command(
            update={
                "review_decision": decision,
                "feedback": decision.get("feedback", ""),
                "validation_errors": [],
                "status": "running",
            },
            goto=replan_target,
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
    builder.add_node("daily_planner", daily_planner_node)
    builder.add_node("validate_plan", validate_plan_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("finalize", finalize_node)
    builder.add_node("failed", failed_node)
    builder.add_node("cancelled", cancelled_node)

    builder.add_edge(START, "normalize_request")
    builder.add_edge("normalize_request", "search")
    builder.add_edge("normalize_request", "weather")
    builder.add_edge(["search", "weather"], "hotel")
    builder.add_conditional_edges(
        "hotel",
        route_after_hotel,
        {
            "planner": "planner",
            "daily_planner": "daily_planner",
        },
    )
    builder.add_edge("planner", "validate_plan")
    builder.add_edge("daily_planner", "validate_plan")

    builder.add_conditional_edges(
        "validate_plan",
        route_after_validation,
        {
            "planner": "planner",
            "daily_planner": "daily_planner",
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
