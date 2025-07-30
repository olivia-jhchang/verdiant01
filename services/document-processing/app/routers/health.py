"""
헬스체크 라우터
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from app.processors.document_processor import DocumentProcessor
from app.main import get_document_processor

router = APIRouter()


@router.get("/health")
async def health_check(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """서비스 헬스체크"""
    try:
        stats = processor.get_processing_stats()
        
        return {
            "service": "document-processing",
            "status": "healthy" if processor.is_initialized else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "processing_stats": stats,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "service": "document-processing",
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }


@router.get("/status")
async def get_status(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """상세 상태 정보"""
    try:
        processing_stats = processor.get_processing_stats()
        chunking_stats = processor.korean_chunker.get_chunking_stats()
        
        return {
            "processing_stats": processing_stats,
            "chunking_stats": chunking_stats,
            "is_initialized": processor.is_initialized,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }