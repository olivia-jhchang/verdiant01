"""
데이터 수집 라우터
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.database.manager import DataCollectionManager
from app.main import get_db_manager
from app.models.requests import (
    CollectionRequest, 
    CollectionResponse, 
    TableAnalysisRequest,
    TableInfo,
    ConnectionStatus,
    ExtractionStats
)
from shared.utils.logging import setup_logging

logger = setup_logging("collection-router")

router = APIRouter()


@router.get("/tables")
async def get_available_tables(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """사용 가능한 테이블 목록 조회"""
    try:
        tables = await db_manager.get_available_tables()
        return {
            "success": True,
            "tables": tables,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"테이블 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테이블 목록 조회 실패: {e}")


@router.post("/collect", response_model=CollectionResponse)
async def collect_documents(
    request: CollectionRequest,
    background_tasks: BackgroundTasks,
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """문서 수집 실행"""
    try:
        logger.info(f"문서 수집 요청: {request}")
        
        # 문서 수집 실행
        result = await db_manager.collect_documents(
            table_names=request.table_names,
            incremental=request.incremental,
            limit=request.limit
        )
        
        response = CollectionResponse(
            success=result["success"],
            message="문서 수집이 완료되었습니다" if result["success"] else "문서 수집 중 오류가 발생했습니다",
            total_documents=result["total_documents"],
            processed_documents=result["processed_documents"],
            failed_documents=result["failed_documents"],
            processing_time=result["processing_time"],
            tables_processed=result["tables_processed"],
            errors=result["errors"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"문서 수집 실패: {e}")
        raise HTTPException(status_code=500, detail=f"문서 수집 실패: {e}")


@router.get("/collect/status")
async def get_collection_status(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """수집 상태 조회"""
    try:
        status = await db_manager.get_connection_status()
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"수집 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"수집 상태 조회 실패: {e}")


@router.post("/test-connection")
async def test_database_connection(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """데이터베이스 연결 테스트"""
    try:
        success = await db_manager.test_connection()
        return {
            "success": success,
            "message": "데이터베이스 연결 성공" if success else "데이터베이스 연결 실패",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"연결 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"연결 테스트 실패: {e}")


@router.get("/schema/{table_name}")
async def get_table_schema(
    table_name: str,
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """특정 테이블의 스키마 조회"""
    try:
        # 사용 가능한 테이블 목록에서 해당 테이블 찾기
        available_tables = await db_manager.get_available_tables()
        table_info = next(
            (table for table in available_tables if table["table_name"] == table_name),
            None
        )
        
        if not table_info:
            raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
        
        return {
            "success": True,
            "table_info": table_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"테이블 스키마 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테이블 스키마 조회 실패: {e}")


@router.post("/analyze-schema/{table_name}")
async def analyze_table_schema(
    table_name: str,
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """테이블 스키마 상세 분석"""
    try:
        # 스키마 분석 실행
        available_tables = await db_manager.get_available_tables()
        table_info = next(
            (table for table in available_tables if table["table_name"] == table_name),
            None
        )
        
        if not table_info:
            raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
        
        schema_analysis = table_info.get('schema_analysis', {})
        
        return {
            "success": True,
            "table_name": table_name,
            "analysis": schema_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"스키마 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"스키마 분석 실패: {e}")


@router.get("/extraction-stats")
async def get_extraction_stats(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """문서 추출 통계 조회"""
    try:
        extractor_stats = db_manager.document_extractor.get_extraction_stats()
        connection_status = await db_manager.get_connection_status()
        
        return {
            "success": True,
            "extraction_stats": extractor_stats,
            "connection_stats": connection_status["stats"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"추출 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"추출 통계 조회 실패: {e}")


@router.post("/reset-stats")
async def reset_extraction_stats(
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """추출 통계 초기화"""
    try:
        db_manager.document_extractor.reset_stats()
        db_manager.schema_detector.clear_cache()
        
        return {
            "success": True,
            "message": "추출 통계가 초기화되었습니다",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 초기화 실패: {e}")


@router.get("/preview/{table_name}")
async def preview_table_data(
    table_name: str,
    limit: int = 5,
    db_manager: DataCollectionManager = Depends(get_db_manager)
):
    """테이블 데이터 미리보기"""
    try:
        # 테이블 존재 확인
        available_tables = await db_manager.get_available_tables()
        table_exists = any(
            table["table_name"] == table_name for table in available_tables
        )
        
        if not table_exists:
            raise HTTPException(status_code=404, detail=f"테이블 '{table_name}'을 찾을 수 없습니다")
        
        # 샘플 데이터 조회
        import asyncio
        loop = asyncio.get_event_loop()
        
        sample_data = await loop.run_in_executor(
            None,
            db_manager.db_manager.execute_query,
            f"SELECT * FROM {table_name} LIMIT {limit}"
        )
        
        return {
            "success": True,
            "table_name": table_name,
            "sample_data": sample_data,
            "sample_count": len(sample_data),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"테이블 미리보기 실패: {e}")
        raise HTTPException(status_code=500, detail=f"테이블 미리보기 실패: {e}")