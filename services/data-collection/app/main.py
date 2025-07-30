"""
데이터 수집 서비스 메인 애플리케이션
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager

from shared.models.config import DatabaseConfig, ServiceConfig
from shared.utils.logging import setup_logging
from app.database.manager import DataCollectionManager
from app.routers import collection, health
from app.middleware.error_handler import ErrorHandlerMiddleware

# 로깅 설정
logger = setup_logging("data-collection")

# 전역 변수
db_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 생명주기 관리"""
    global db_manager
    
    # 시작 시 초기화
    logger.info("데이터 수집 서비스 시작")
    
    try:
        # 데이터베이스 설정 로드
        db_config = DatabaseConfig.from_env()
        db_manager = DataCollectionManager(db_config)
        
        # 데이터베이스 연결 테스트
        await db_manager.test_connection()
        logger.info("데이터베이스 연결 성공")
        
        # 애플리케이션 상태에 매니저 저장
        app.state.db_manager = db_manager
        
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}")
        raise e
    
    yield
    
    # 종료 시 정리
    logger.info("데이터 수집 서비스 종료")
    if db_manager:
        await db_manager.close()


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="데이터 수집 서비스",
    description="내부 데이터베이스에서 문서를 수집하는 서비스",
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
app.include_router(collection.router, prefix="/api/v1", tags=["collection"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "데이터 수집 서비스",
        "status": "running",
        "version": "1.0.0"
    }


def get_db_manager() -> DataCollectionManager:
    """데이터베이스 매니저 의존성"""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="데이터베이스 매니저가 초기화되지 않았습니다")
    return db_manager


if __name__ == "__main__":
    service_config = ServiceConfig.from_env()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=service_config.data_collection_port,
        reload=True
    )