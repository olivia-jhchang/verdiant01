"""
헬스체크 라우터
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from app.database.manager import DataCollectionManager
from app.main import get_db_manager

router = APIRouter()


@router.get("/health")
async def health_check(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """서비스 헬스체크"""
    try:
        status = await db_manager.get_connection_status()
        
        return {
            "service": "data-collection",
            "status": "healthy" if status["status"] == "connected" else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "database": status,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "service": "data-collection",
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }


@router.get("/status")
async def get_status(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """상세 상태 정보"""
    try:
        connection_status = await db_manager.get_connection_status()
        available_tables = await db_manager.get_available_tables()
        
        return {
            "connection": connection_status,
            "tables": available_tables,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }