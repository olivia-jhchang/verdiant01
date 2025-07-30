"""
검색 서비스 메인 애플리케이션
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from shared.models.config import EmbeddingConfig, VectorDBConfig, LLMConfig, ServiceConfig
from shared.utils.logging import setup_logging
from app.search.search_engine import SearchEngine
from app.llm.response_generator import ResponseGenerator
from app.routers import search, health
from app.middleware.error_handler import ErrorHandlerMiddleware

# 로깅 설정
logger = setup_logging("search-service")

# 전역 변수
search_engine = None
response_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global search_engine, response_generator
    
    # 시작 시 초기화
    logger.info("검색 서비스 시작")
    
    try:
        # 설정 로드
        embedding_config = EmbeddingConfig.from_env()
        vectordb_config = VectorDBConfig.from_env()
        llm_config = LLMConfig.from_env()
        
        # 검색 엔진 초기화
        search_engine = SearchEngine(embedding_config, vectordb_config)
        await search_engine.initialize()
        
        # 응답 생성기 초기화
        response_generator = ResponseGenerator(llm_config)
        await response_generator.initialize()
        
        logger.info("검색 서비스 초기화 완료")
        
        # 애플리케이션 상태에 저장
        app.state.search_engine = search_engine
        app.state.response_generator = response_generator
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise e
    
    yield
    
    # 종료 시 정리
    logger.info("검색 서비스 종료")
    if search_engine:
        await search_engine.cleanup()
    if response_generator:
        await response_generator.cleanup()


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="검색 서비스",
    description="RAG 기반 지능형 검색 및 응답 생성 서비스",
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
app.include_router(search.router, prefix="/api/v1", tags=["search"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "검색 서비스",
        "status": "running",
        "version": "1.0.0"
    }


def get_search_engine() -> SearchEngine:
    """검색 엔진 의존성"""
    if search_engine is None:
        raise HTTPException(status_code=500, detail="검색 엔진이 초기화되지 않았습니다")
    return search_engine


def get_response_generator() -> ResponseGenerator:
    """응답 생성기 의존성"""
    if response_generator is None:
        raise HTTPException(status_code=500, detail="응답 생성기가 초기화되지 않았습니다")
    return response_generator


if __name__ == "__main__":
    service_config = ServiceConfig.from_env()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=service_config.search_service_port,
        reload=True
    )