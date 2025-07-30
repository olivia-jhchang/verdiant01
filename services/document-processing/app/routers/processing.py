"""
문서 처리 라우터
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.processors.document_processor import DocumentProcessor
from app.main import get_document_processor
from app.models.requests import (
    ProcessingRequest,
    ProcessingResponse,
    BatchProcessingRequest,
    BatchProcessingResponse,
    ChunkingRequest,
    ChunkingResponse
)
from shared.models.document import Document, DocumentType
from shared.utils.logging import setup_logging

logger = setup_logging("processing-router")

router = APIRouter()


@router.post("/process", response_model=ProcessingResponse)
async def process_document(
    request: ProcessingRequest,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """단일 문서 처리"""
    try:
        logger.info(f"문서 처리 요청: {request.document.id}")
        
        # 문서 처리 실행
        chunks = await processor.process_document(request.document)
        
        response = ProcessingResponse(
            success=True,
            message="문서 처리가 완료되었습니다",
            document_id=request.document.id,
            chunks_created=len(chunks),
            chunks=chunks,
            processing_time=0.0  # 실제로는 processor에서 측정
        )
        
        return response
        
    except Exception as e:
        logger.error(f"문서 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"문서 처리 실패: {e}")


@router.post("/process-batch", response_model=BatchProcessingResponse)
async def process_documents_batch(
    request: BatchProcessingRequest,
    background_tasks: BackgroundTasks,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """문서 배치 처리"""
    try:
        logger.info(f"배치 문서 처리 요청: {len(request.documents)}개 문서")
        
        # 배치 처리 실행
        results = await processor.process_documents_batch(request.documents)
        
        # 결과 집계
        total_chunks = sum(len(chunks) for chunks in results)
        successful_documents = len([r for r in results if r])
        failed_documents = len(request.documents) - successful_documents
        
        response = BatchProcessingResponse(
            success=True,
            message="배치 문서 처리가 완료되었습니다",
            total_documents=len(request.documents),
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            total_chunks_created=total_chunks,
            results=results,
            processing_time=0.0
        )
        
        return response
        
    except Exception as e:
        logger.error(f"배치 문서 처리 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 문서 처리 실패: {e}")


@router.post("/chunk", response_model=ChunkingResponse)
async def chunk_document(
    request: ChunkingRequest,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """문서 청킹만 수행"""
    try:
        logger.info(f"문서 청킹 요청: {request.document.id}")
        
        # 구조 분석
        structure_info = await processor._analyze_document_structure(request.document)
        
        # 텍스트 정리
        cleaned_content = await processor._clean_and_normalize_text(request.document.content)
        
        # 청킹 수행
        chunks = await processor.korean_chunker.chunk_document(
            request.document,
            cleaned_content,
            structure_info,
            request.strategy
        )
        
        response = ChunkingResponse(
            success=True,
            message="문서 청킹이 완료되었습니다",
            document_id=request.document.id,
            strategy_used=request.strategy or "auto",
            chunks_created=len(chunks),
            chunks=chunks,
            structure_info=structure_info
        )
        
        return response
        
    except Exception as e:
        logger.error(f"문서 청킹 실패: {e}")
        raise HTTPException(status_code=500, detail=f"문서 청킹 실패: {e}")


@router.get("/stats")
async def get_processing_stats(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """처리 통계 조회"""
    try:
        processing_stats = processor.get_processing_stats()
        chunking_stats = processor.korean_chunker.get_chunking_stats()
        
        return {
            "success": True,
            "processing_stats": processing_stats,
            "chunking_stats": chunking_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {e}")


@router.post("/reset-stats")
async def reset_processing_stats(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """처리 통계 초기화"""
    try:
        processor.reset_stats()
        processor.korean_chunker.reset_stats()
        
        return {
            "success": True,
            "message": "처리 통계가 초기화되었습니다",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 초기화 실패: {e}")


@router.post("/analyze-structure")
async def analyze_document_structure(
    request: ProcessingRequest,
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """문서 구조 분석"""
    try:
        logger.info(f"문서 구조 분석 요청: {request.document.id}")
        
        # 구조 분석 수행
        structure_info = await processor._analyze_document_structure(request.document)
        
        # 문서 분류 재검증
        refined_type = await processor._refine_document_classification(request.document)
        
        return {
            "success": True,
            "document_id": request.document.id,
            "original_type": request.document.document_type.value,
            "refined_type": refined_type.value,
            "structure_info": structure_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"구조 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"구조 분석 실패: {e}")


@router.get("/strategies")
async def get_available_strategies(
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """사용 가능한 청킹 전략 조회"""
    try:
        strategies = list(processor.korean_chunker.chunkers.keys())
        strategy_stats = processor.korean_chunker.chunking_stats.get("strategy_usage", {})
        
        return {
            "success": True,
            "available_strategies": strategies,
            "strategy_usage_stats": strategy_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"전략 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"전략 조회 실패: {e}")


@router.post("/test-chunking")
async def test_chunking_strategy(
    text: str,
    strategy: str,
    document_type: str = "administrative",
    processor: DocumentProcessor = Depends(get_document_processor)
):
    """청킹 전략 테스트"""
    try:
        # 테스트용 문서 생성
        test_document = Document(
            id="test_doc",
            title="테스트 문서",
            content=text,
            document_type=DocumentType(document_type),
            source_table="test",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            metadata={}
        )
        
        # 구조 분석
        structure_info = await processor._analyze_document_structure(test_document)
        
        # 청킹 수행
        chunks = await processor.korean_chunker.chunk_document(
            test_document,
            text,
            structure_info,
            strategy
        )
        
        return {
            "success": True,
            "strategy_used": strategy,
            "chunks_created": len(chunks),
            "chunks": [{"text": chunk.text, "metadata": chunk.metadata} for chunk in chunks],
            "structure_info": structure_info,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"청킹 테스트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"청킹 테스트 실패: {e}")