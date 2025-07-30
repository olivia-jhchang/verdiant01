"""
문서 처리 서비스 메인 애플리케이션
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from shared.models.config import ChunkingConfig, ServiceConfig
from shared.utils.logging import setup_logging
from app.processors.document_processor import DocumentProcessor
from app.routers import processing, health
from app.middleware.error_handler import ErrorHandlerMiddleware

# 로깅 설정
logger = setup_logging("document-processing")

# 전역 변수
document_processor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global document_processor
    
    # 시작 시 초기화
    logger.info("문서 처리 서비스 시작")
    
    try:
        # 청킹 설정 로드
        chunking_config = ChunkingConfig.from_env()
        document_processor = DocumentProcessor(chunking_config)
        
        # 한국어 처리 모델 초기화
        await document_processor.initialize()
        logger.info("문서 처리기 초기화 완료")
        
        # 애플리케이션 상태에 프로세서 저장
        app.state.document_processor = document_processor
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise e
    
    yield
    
    # 종료 시 정리
    logger.info("문서 처리 서비스 종료")
    if document_processor:
        await document_processor.cleanup()


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="문서 처리 서비스",
    description="한국어 문서 구조화 및 청킹 서비스",
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
app.include_router(processing.router, prefix="/api/v1", tags=["processing"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "문서 처리 서비스",
        "status": "running",
        "version": "1.0.0"
    }


def get_document_processor() -> DocumentProcessor:
    """문서 처리기 의존성"""
    if document_processor is None:
        raise HTTPException(status_code=500, detail="문서 처리기가 초기화되지 않았습니다")
    return document_processor


if __name__ == "__main__":
    service_config = ServiceConfig.from_env()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=service_config.document_processing_port,
        reload=True
    )