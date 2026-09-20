from fastapi import APIRouter
from app.services.tool_registry import list_tools

router = APIRouter()


@router.get("/api/tools")
def get_tools():
    return [
        {
            "name": t.name,
            "task": t.task,
            "required_mode": t.required_mode,
            "engine": t.engine,
            "status": t.status,
        }
        for t in list_tools()
    ]
