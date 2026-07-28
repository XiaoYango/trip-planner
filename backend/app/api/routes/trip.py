"""旅行规划API路由"""

from uuid import uuid4

from fastapi import APIRouter, HTTPException
from langgraph.types import Command

from ...graph.trip_graph import get_trip_graph
from ...models.schemas import (
    TripRequest,
    TripReviewRequest,
    TripWorkflowResponse,
)
from ...agents.trip_planner_agent import get_trip_planner_agent

router = APIRouter(prefix="/trip", tags=["旅行规划"])


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
    messages = {
        "completed": "旅行计划已确认",
        "cancelled": "用户已取消旅行计划",
        "failed": "旅行计划生成失败",
    }
    return TripWorkflowResponse(
        success=status == "completed",
        status=status,
        thread_id=thread_id,
        message=messages.get(status, "旅行规划状态未知"),
        data=result.get("final_plan"),
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
        print(f"❌ 生成旅行计划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"生成旅行计划失败: {str(e)}"
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
        print(f"❌ 恢复旅行规划失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"恢复旅行规划失败: {str(e)}"
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
