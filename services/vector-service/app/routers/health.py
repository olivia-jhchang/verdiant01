"""
헬스체크 라우터
"""
from fastapi import APIRouter, Depends
from datetime import datetime
from app.embeddings.embedding_manager import EmbeddingManager
from app.indexing.vector_indexer import VectorIndexer
from app.main import get_embedding_manager, get_vector_indexer

router = APIRouter()


@router.get("/health")
async def health_check(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """서비스 헬스체크"""
    try:
        embedding_stats = embedding_manager.get_embedding_stats()
        indexing_stats = vector_indexer.get_indexing_stats()
        
        return {
            "service": "vector-service",
            "status": "healthy" if (embedding_manager.is_initialized and vector_indexer.is_initialized) else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "embedding_stats": embedding_stats,
            "indexing_stats": indexing_stats,
            "version": "1.0.0"
        }
    except Exception as e:
        return {
            "service": "vector-service",
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "version": "1.0.0"
        }


@router.get("/status")
async def get_status(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """상세 상태 정보"""
    try:
        embedding_stats = embedding_manager.get_embedding_stats()
        indexing_stats = vector_indexer.get_indexing_stats()
        collection_names = await vector_indexer.get_collection_names()
        
        return {
            "embedding_manager": {
                "is_initialized": embedding_manager.is_initialized,
                "stats": embedding_stats
            },
            "vector_indexer": {
                "is_initialized": vector_indexer.is_initialized,
                "stats": indexing_stats,
                "collections": collection_names
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }