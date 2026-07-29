"""旅行规划API路由"""

import json
import re
from collections.abc import AsyncIterator
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from ...graph.trip_graph import get_trip_graph
from ...models.schemas import (
    TripRequest,
    TripReviewRequest,
    TripWorkflowResponse,
)
from ...agents.trip_planner_agent import get_trip_planner_agent
from ...config import get_settings

router = APIRouter(prefix="/trip", tags=["旅行规划"])

_API_KEY_PATTERN = re.compile(
    r"\b(?:sk|ak|key)-[A-Za-z0-9_*.-]{8,}\b",
    flags=re.IGNORECASE,
)


def _format_workflow_error(error: object) -> str:
    """对内部异常脱敏，并转换为适合工作台展示的错误原因。"""
    raw = str(error)
    settings = get_settings()
    for secret in (
        settings.llm_api_key,
        settings.amap_api_key,
        settings.unsplash_access_key,
        settings.unsplash_secret_key,
    ):
        if secret:
            raw = raw.replace(secret, "[已隐藏的API密钥]")
    raw = _API_KEY_PATTERN.sub("[已隐藏的API密钥]", raw)
    lowered = raw.lower()
    source = raw.split("：", 1)[0] if "：" in raw else "旅行规划工作流"

    if "invalid_api_key" in lowered or "incorrect api key" in lowered or "401" in lowered:
        return (
            f"{source}：LLM鉴权失败（401）。"
            "请检查LLM_API_KEY是否有效，并确认LLM_BASE_URL与密钥属于同一服务商。"
        )
    if "model_not_found" in lowered or ("404" in lowered and "model" in lowered):
        return (
            f"{source}：模型不可用（404）。"
            "请检查LLM_MODEL_ID是否存在，以及当前账号是否有调用权限。"
        )
    if "rate_limit" in lowered or "429" in lowered:
        return f"{source}：LLM请求受到限流（429），请稍后重试或检查账户额度。"
    if "timeout" in lowered or "timed out" in lowered:
        return f"{source}：外部服务请求超时，请检查网络、代理或服务状态。"

    return raw


def _encode_sse(event: str, data: dict) -> str:
    """编码单个Server-Sent Event。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _build_workflow_response(
    result: dict,
    thread_id: str,
) -> TripWorkflowResponse:
    """将LangGraph执行结果转换为API响应。"""
    interrupts = result.get("__interrupt__", ())

    if interrupts:
        review = getattr(interrupts[0], "value", interrupts[0])
        return TripWorkflowResponse(
            success=True,
            status="awaiting_approval",
            thread_id=thread_id,
            message="旅行计划草案已生成，等待用户确认",
            data=result.get("draft"),
            review=review,
        )

    status = result.get("status", "failed")
    errors = [
        _format_workflow_error(error)
        for error in result.get("validation_errors", [])
    ]
    messages = {
        "completed": "旅行计划已确认",
        "cancelled": "用户已取消旅行计划",
        "failed": "旅行计划生成失败，请查看调试信息",
    }
    return TripWorkflowResponse(
        success=status == "completed",
        status=status,
        thread_id=thread_id,
        message=messages.get(status, "旅行规划状态未知"),
        data=result.get("final_plan"),
        errors=errors,
        attempts=result.get("attempts", 0),
    )


@router.post(
    "/plan",
    response_model=TripWorkflowResponse,
    summary="生成旅行计划",
    description="生成旅行计划草案，并在人工确认节点暂停"
)
async def plan_trip(request: TripRequest):
    """
    生成旅行计划

    Args:
        request: 旅行请求参数

    Returns:
        旅行计划响应
    """
    try:
        print(f"\n{'='*60}")
        print(f"📥 收到旅行规划请求:")
        print(f"   城市: {request.city}")
        print(f"   日期: {request.start_date} - {request.end_date}")
        print(f"   天数: {request.travel_days}")
        print(f"{'='*60}\n")

        graph = get_trip_graph()
        thread_id = str(uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        print("🚀 开始执行LangGraph旅行规划状态图...")
        result = await graph.ainvoke(
            {
                "request": request.model_dump(),
                "status": "running",
                "attempts": 0,
                "validation_errors": [],
                "feedback": "",
            },
            config=config,
        )

        return _build_workflow_response(result, thread_id)

    except Exception as e:
        safe_error = _format_workflow_error(e)
        print(f"❌ 生成旅行计划失败: {safe_error}")
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {safe_error}"
    )


async def _stream_daily_plan(
    request: TripRequest,
    thread_id: str,
) -> AsyncIterator[str]:
    """执行逐日LangGraph，并把状态、每日结果及最终审核状态推送给前端。"""
    graph = get_trip_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "request": request.model_dump(),
        "status": "running",
        "attempts": 0,
        "validation_errors": [],
        "feedback": "",
        "generation_mode": "daily",
    }
    streamed_interrupts = ()

    yield _encode_sse(
        "status",
        {
            "node": "initialize",
            "message": "旅行规划任务已创建",
            "progress": 2,
            "thread_id": thread_id,
            "total_days": request.travel_days,
        },
    )

    try:
        async for stream_item in graph.astream(
            initial_state,
            config=config,
            stream_mode=["custom", "updates"],
        ):
            if (
                isinstance(stream_item, tuple)
                and len(stream_item) == 2
                and stream_item[0] in {"custom", "updates"}
            ):
                stream_mode, payload = stream_item
            else:
                stream_mode, payload = "updates", stream_item

            if stream_mode == "custom" and isinstance(payload, dict):
                event = str(payload.get("event", "status"))
                event_payload = dict(payload)
                event_payload.pop("event", None)
                event_payload.setdefault("thread_id", thread_id)
                yield _encode_sse(event, event_payload)
                continue

            if isinstance(payload, dict) and "__interrupt__" in payload:
                streamed_interrupts = payload["__interrupt__"]

        snapshot = await graph.aget_state(config)
        result = dict(snapshot.values)

        if not streamed_interrupts:
            streamed_interrupts = tuple(
                item
                for task in snapshot.tasks
                for item in getattr(task, "interrupts", ())
            )
        if streamed_interrupts:
            result["__interrupt__"] = streamed_interrupts

        response = _build_workflow_response(result, thread_id)
        response_data = response.model_dump(mode="json")

        if response.status == "awaiting_approval":
            yield _encode_sse("review", response_data)
        elif response.status == "failed":
            yield _encode_sse("error", response_data)
        else:
            yield _encode_sse("complete", response_data)
    except Exception as exc:
        safe_error = _format_workflow_error(exc)
        yield _encode_sse(
            "error",
            {
                "success": False,
                "status": "failed",
                "thread_id": thread_id,
                "message": "流式旅行计划生成失败",
                "errors": [safe_error],
                "attempts": 0,
            },
        )


@router.post(
    "/plan/stream",
    summary="按天流式生成旅行计划",
    description="通过SSE逐步返回节点状态、每日计划和最终人工审核状态",
)
async def stream_trip_plan(request: TripRequest):
    """新增流式入口；原有/plan接口保持不变。"""
    thread_id = str(uuid4())
    return StreamingResponse(
        _stream_daily_plan(request, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/plan/{thread_id}/resume",
    response_model=TripWorkflowResponse,
    summary="恢复旅行规划工作流",
    description="提交人工审核结果并从中断点恢复旅行规划"
)
async def resume_trip_plan(
    thread_id: str,
    review: TripReviewRequest,
):
    """恢复等待人工确认的旅行规划状态图。"""
    try:
        graph = get_trip_graph()
        config = {"configurable": {"thread_id": thread_id}}
        result = await graph.ainvoke(
            Command(resume=review.model_dump(exclude_none=True)),
            config=config,
        )
        return _build_workflow_response(result, thread_id)
    except Exception as e:
        safe_error = _format_workflow_error(e)
        print(f"❌ 恢复旅行规划失败: {safe_error}")
        raise HTTPException(
            status_code=500,
            detail=f"恢复旅行规划失败: {safe_error}"
        )


@router.get(
    "/health",
    summary="健康检查",
    description="检查旅行规划服务是否正常"
)
async def health_check():
    """健康检查"""
    try:
        # 检查Agent是否可用
        agent = get_trip_planner_agent()
        
        return {
            "status": "healthy",
            "service": "trip-planner",
            "agent_name": agent.agent.name,
            "tools_count": len(agent.agent.list_tools())
        }
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"服务不可用: {str(e)}"
        )
