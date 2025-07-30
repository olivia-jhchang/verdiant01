"""
벡터 서비스 메인 애플리케이션
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from shared.models.config import EmbeddingConfig, VectorDBConfig, ServiceConfig
from shared.utils.logging import setup_logging
from app.embeddings.embedding_manager import EmbeddingManager
from app.indexing.vector_indexer import VectorIndexer
from app.routers import vectorization, health
from app.middleware.error_handler import ErrorHandlerMiddleware

# 로깅 설정
logger = setup_logging("vector-service")

# 전역 변수
embedding_manager = None
vector_indexer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global embedding_manager, vector_indexer
    
    # 시작 시 초기화
    logger.info("벡터 서비스 시작")
    
    try:
        # 설정 로드
        embedding_config = EmbeddingConfig.from_env()
        vectordb_config = VectorDBConfig.from_env()
        
        # 임베딩 매니저 초기화
        embedding_manager = EmbeddingManager(embedding_config)
        await embedding_manager.initialize()
        
        # 벡터 인덱서 초기화
        vector_indexer = VectorIndexer(vectordb_config)
        await vector_indexer.initialize()
        
        logger.info("벡터 서비스 초기화 완료")
        
        # 애플리케이션 상태에 매니저들 저장
        app.state.embedding_manager = embedding_manager
        app.state.vector_indexer = vector_indexer
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise e
    
    yield
    
    # 종료 시 정리
    logger.info("벡터 서비스 종료")
    if embedding_manager:
        await embedding_manager.cleanup()
    if vector_indexer:
        await vector_indexer.cleanup()


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="벡터 서비스",
    description="문서 임베딩 생성 및 벡터 인덱싱 서비스",
    version="1.0.0",
    lifespan=lifespan
)

# 미들웨어 설정
app.add_middleware(ErrorHandlerMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(vectorization.router, prefix="/api/v1", tags=["vectorization"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "벡터 서비스",
        "status": "running",
        "version": "1.0.0"
    }


def get_embedding_manager() -> EmbeddingManager:
    """임베딩 매니저 의존성"""
    if embedding_manager is None:
        raise HTTPException(status_code=500, detail="임베딩 매니저가 초기화되지 않았습니다")
    return embedding_manager


def get_vector_indexer() -> VectorIndexer:
    """벡터 인덱서 의존성"""
    if vector_indexer is None:
        raise HTTPException(status_code=500, detail="벡터 인덱서가 초기화되지 않았습니다")
    return vector_indexer


if __name__ == "__main__":
    service_config = ServiceConfig.from_env()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=service_config.vector_service_port,
        reload=True
    )