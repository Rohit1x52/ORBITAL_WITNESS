from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
from datetime import datetime

from ..models.requests import AnalysisRequest, ClassificationRequest
from ..models.responses import AnalysisResponse, ClassificationResponse, TaskResponse
from ..config import settings
from ..graph_runtime import get_workflow_runner

router = APIRouter()
logger = logging.getLogger(__name__)

tasks_db: Dict[str, Dict[str, Any]] = {}

def _run_analysis(input_data: Dict[str, Any], task_id: str | None = None) -> Dict[str, Any]:
    runner = get_workflow_runner()
    if settings.WORKFLOW_MODE.lower() == "linear":
        return runner.invoke(input_data)
    return runner.invoke(input_data, thread_id=task_id)

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_satellite_data(request: AnalysisRequest):
    try:
        logger.info(f"Received analysis request for location: {request.location}")
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        
        result = _run_analysis(input_data)
        
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
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@router.post("/analyze/async", response_model=TaskResponse)
async def analyze_satellite_data_async(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    task_id = f"task_{datetime.now().timestamp()}"
    
    tasks_db[task_id] = {
        "status": "processing",
        "created_at": datetime.now().isoformat(),
        "request": request.dict(),
        "workflow_mode": settings.WORKFLOW_MODE,
        "graph_run_id": task_id if settings.WORKFLOW_MODE.lower() == "graph" else None,
    }
    
    async def process_analysis():
        try:
            input_data = {
                "location": request.location,
                "before_date": request.before_date,
                "after_date": request.after_date
            }
            result = _run_analysis(input_data, task_id=task_id)
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = result
        except Exception as e:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error"] = str(e)
    
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
async def classify_image_change(request: ClassificationRequest):
    try:
        logger.info(f"Received classification request for location: {request.location}")
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        result = _run_analysis(input_data)
        
        return ClassificationResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            location=request.location,
            classification=result.get("classification", {}),
            summary=result.get("summary", "")
        )
    
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )