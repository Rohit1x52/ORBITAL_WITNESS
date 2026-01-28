from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
import logging
from datetime import datetime

from ..models.requests import AnalysisRequest, ClassificationRequest
from ..models.responses import AnalysisResponse, ClassificationResponse, TaskResponse

router = APIRouter()
logger = logging.getLogger(__name__)

tasks_db: Dict[str, Dict[str, Any]] = {}
agent_instance = None
_agent_lock = False

def get_agent():
    global agent_instance, _agent_lock
    
    if agent_instance is None:
        if not _agent_lock:
            _agent_lock = True
            logger.info("Initializing agent on first request...")
            from app.agent import create_satellite_agent
            agent_instance = create_satellite_agent()
            logger.info("Agent initialized successfully")
        else:
            raise HTTPException(status_code=503, detail="Agent is being initialized, please retry")
    
    return agent_instance

def set_agent(agent):
    global agent_instance
    agent_instance = agent

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_satellite_data(request: AnalysisRequest):
    try:
        agent = get_agent()
        logger.info(f"Received analysis request for location: {request.location}")
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        
        result = agent.invoke(input_data)
        
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
        "request": request.dict()
    }
    
    async def process_analysis():
        try:
            agent = get_agent()
            input_data = {
                "location": request.location,
                "before_date": request.before_date,
                "after_date": request.after_date
            }
            result = agent.invoke(input_data)
            tasks_db[task_id]["status"] = "completed"
            tasks_db[task_id]["result"] = result
        except Exception as e:
            tasks_db[task_id]["status"] = "failed"
            tasks_db[task_id]["error"] = str(e)
    
    background_tasks.add_task(process_analysis)
    
    return TaskResponse(
        task_id=task_id,
        status="processing",
        message="Analysis task submitted successfully"
    )

@router.get("/analyze/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return tasks_db[task_id]

@router.post("/classify", response_model=ClassificationResponse)
async def classify_image_change(request: ClassificationRequest):
    try:
        agent = get_agent()
        logger.info(f"Received classification request for location: {request.location}")
        
        input_data = {
            "location": request.location,
            "before_date": request.before_date,
            "after_date": request.after_date
        }
        
        data = agent.fetch_satellite_data(input_data)
        result = agent.run_classification(data)
        
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