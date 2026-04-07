from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request, Query
from typing import Dict, Any
import logging
from datetime import datetime
from uuid import uuid4

from sqlalchemy.orm import Session

from ..models.requests import AnalysisRequest, ClassificationRequest
from ..models.responses import AnalysisResponse, ClassificationResponse, TaskResponse
from ..config import settings
from ..graph_runtime import get_workflow_runner
from ..db.session import get_db, SessionLocal
from ..db import crud

router = APIRouter()
logger = logging.getLogger(__name__)

tasks_db: Dict[str, Dict[str, Any]] = {}

def _run_analysis(input_data: Dict[str, Any], task_id: str | None = None) -> Dict[str, Any]:
    runner = get_workflow_runner()
    if settings.WORKFLOW_MODE.lower() == "linear":
        return runner.invoke(input_data)
    return runner.invoke(input_data, thread_id=task_id)


def _extract_client_context(request: Request) -> Dict[str, str | None]:
    session_id = request.headers.get("x-session-id") or f"session_{uuid4().hex}"
    external_user_id = request.headers.get("x-user-id")
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    return {
        "session_id": session_id,
        "external_user_id": external_user_id,
        "user_agent": user_agent,
        "ip_address": ip_address,
    }


def _persist_analysis(
    db: Session,
    *,
    context: Dict[str, str | None],
    request_payload: AnalysisRequest,
    result: Dict[str, Any],
    task_id: str | None,
    status: str,
) -> None:
    user = crud.get_or_create_user(db, external_id=context.get("external_user_id"))
    crud.get_or_create_session(
        db,
        session_id=context["session_id"] or "unknown",
        user_agent=context.get("user_agent"),
        ip_address=context.get("ip_address"),
        user=user,
    )
    crud.create_analysis_history(
        db,
        task_id=task_id,
        session_id=context.get("session_id"),
        user_id=user.id if user else None,
        location=request_payload.location,
        before_date=request_payload.before_date,
        after_date=request_payload.after_date,
        workflow_mode=settings.WORKFLOW_MODE,
        classification=result.get("classification", {}),
        summary=result.get("summary", ""),
        solutions=result.get("solutions", ""),
        status=status,
        metadata=result.get("metadata", {}),
    )
    crud.create_audit_log(
        db,
        session_id=context.get("session_id"),
        user_id=user.id if user else None,
        action="analysis_request",
        resource="/api/v1/analyze",
        status=status,
        details={
            "task_id": task_id,
            "workflow_mode": settings.WORKFLOW_MODE,
            "classification": result.get("classification", {}),
        },
    )

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_satellite_data(
    request: AnalysisRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    try:
        logger.info(f"Received analysis request for location: {request.location}")
        context = _extract_client_context(http_request)
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        
        result = _run_analysis(input_data)

        if settings.DATABASE_ENABLED:
            _persist_analysis(
                db,
                context=context,
                request_payload=request,
                result=result,
                task_id=None,
                status="completed",
            )
            db.commit()
        
        return AnalysisResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            location=request.location,
            analysis_period={
                "before": request.before_date,
                "after": request.after_date
            },
            classification=result.get("classification", {}),
            summary=result.get("summary", ""),
            solutions=result.get("solutions", ""),
            metadata={
                "processing_time": result.get("timestamp", ""),
                "confidence": result.get("classification", {}).get("confidence", 0.0)
            }
        )
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        if settings.DATABASE_ENABLED:
            db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/async", response_model=TaskResponse)
async def analyze_satellite_data_async(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
):
    task_id = f"task_{datetime.now().timestamp()}"
    context = _extract_client_context(http_request)
    
    tasks_db[task_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "workflow_mode": settings.WORKFLOW_MODE,
        "graph_run_id": task_id if settings.WORKFLOW_MODE.lower() == "graph" else None,
    }
    
    async def process_analysis():
        db = SessionLocal()
        try:
            input_data = {
                "location": request.location,
                "before_date": request.before_date,
                "after_date": request.after_date
            }
            result = _run_analysis(input_data, task_id=task_id)
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = result
            if settings.DATABASE_ENABLED:
                _persist_analysis(
                    db,
                    context=context,
                    request_payload=request,
                    result=result,
                    task_id=task_id,
                    status="completed",
                )
                db.commit()
        except Exception as e:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error"] = str(e)
            if settings.DATABASE_ENABLED:
                db.rollback()
                crud.create_audit_log(
                    db,
                    session_id=context.get("session_id"),
                    user_id=None,
                    action="analysis_request",
                    resource="/api/v1/analyze/async",
                    status="failed",
                    details={"task_id": task_id, "error": str(e)},
                )
                db.commit()
        finally:
            db.close()
    
    background_tasks.add_task(process_analysis)
    
    return TaskResponse(
        task_id=task_id,
        status="processing",
        message="Analysis task submitted successfully",
        graph_run_id=task_id if settings.WORKFLOW_MODE.lower() == "graph" else None
    )

@router.get("/analyze/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_db[task_id]

@router.post("/classify", response_model=ClassificationResponse)
async def classify_image_change(
    request: ClassificationRequest,
    http_request: Request,
    db: Session = Depends(get_db),
):
    try:
        logger.info(f"Received classification request for location: {request.location}")
        context = _extract_client_context(http_request)
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        result = _run_analysis(input_data)

        if settings.DATABASE_ENABLED:
            user = crud.get_or_create_user(db, external_id=context.get("external_user_id"))
            crud.get_or_create_session(
                db,
                session_id=context["session_id"] or "unknown",
                user_agent=context.get("user_agent"),
                ip_address=context.get("ip_address"),
                user=user,
            )
            crud.create_analysis_history(
                db,
                task_id=None,
                session_id=context.get("session_id"),
                user_id=user.id if user else None,
                location=request.location,
                before_date=request.before_date,
                after_date=request.after_date,
                workflow_mode=settings.WORKFLOW_MODE,
                classification=result.get("classification", {}),
                summary=result.get("summary", ""),
                solutions="",
                status="classified",
                metadata=result.get("metadata", {}),
            )
            crud.create_audit_log(
                db,
                session_id=context.get("session_id"),
                user_id=user.id if user else None,
                action="classify_request",
                resource="/api/v1/classify",
                status="completed",
                details={"classification": result.get("classification", {})},
            )
            db.commit()
        
        return ClassificationResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            location=request.location,
            classification=result.get("classification", {}),
            summary=result.get("summary", "")
        )
    
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        if settings.DATABASE_ENABLED:
            db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.get("/history")
async def get_analysis_history(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    if not settings.DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database layer is disabled")

    rows = crud.list_analysis_history(db, session_id=session_id, limit=limit)
    return {
        "count": len(rows),
        "items": [
            {
                "id": row.id,
                "task_id": row.task_id,
                "session_id": row.session_id,
                "location": (row.location_lat, row.location_lon),
                "analysis_period": {"before": row.before_date, "after": row.after_date},
                "classification_label": row.classification_label,
                "confidence": row.confidence,
                "status": row.status,
                "workflow_mode": row.workflow_mode,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
    }


@router.get("/audit")
async def get_audit_logs(
    session_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    if not settings.DATABASE_ENABLED:
        raise HTTPException(status_code=503, detail="Database layer is disabled")

    rows = crud.list_audit_logs(db, session_id=session_id, limit=limit)
    return {
        "count": len(rows),
        "items": [
            {
                "id": row.id,
                "session_id": row.session_id,
                "action": row.action,
                "resource": row.resource,
                "status": row.status,
                "details": row.details,
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ],
    }