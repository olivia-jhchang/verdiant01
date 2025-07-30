"""
벡터화 라우터
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
from datetime import datetime

from app.embeddings.embedding_manager import EmbeddingManager
from app.embeddings.batch_processor import BatchVectorizationProcessor
from app.indexing.vector_indexer import VectorIndexer
from app.main import get_embedding_manager, get_vector_indexer
from app.models.requests import (
    EmbeddingRequest,
    EmbeddingResponse,
    IndexingRequest,
    IndexingResponse,
    SearchRequest,
    SearchResponse,
    VerificationRequest,
    VerificationResponse
)
from shared.models.document import Chunk
from shared.utils.logging import setup_logging

logger = setup_logging("vectorization-router")

router = APIRouter()


@router.post("/embed", response_model=EmbeddingResponse)
async def embed_texts(
    request: EmbeddingRequest,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """텍스트 임베딩 생성"""
    try:
        logger.info(f"텍스트 임베딩 요청: {len(request.texts)}개 텍스트")
        
        # 임베딩 생성
        embeddings = await embedding_manager.embed_texts(request.texts)
        
        response = EmbeddingResponse(
            success=True,
            message="임베딩 생성이 완료되었습니다",
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            processing_time=0.0  # 실제로는 embedding_manager에서 측정
        )
        
        return response
        
    except Exception as e:
        logger.error(f"텍스트 임베딩 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {e}")


@router.post("/embed-chunks", response_model=EmbeddingResponse)
async def embed_chunks(
    request: IndexingRequest,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager)
):
    """청크 임베딩 생성"""
    try:
        logger.info(f"청크 임베딩 요청: {len(request.chunks)}개 청크")
        
        # 청크 임베딩 생성
        embedded_chunks = await embedding_manager.embed_chunks(request.chunks)
        
        # 임베딩 벡터만 추출
        embeddings = [chunk.vector for chunk in embedded_chunks if chunk.vector]
        
        response = EmbeddingResponse(
            success=True,
            message="청크 임베딩 생성이 완료되었습니다",
            embeddings=embeddings,
            dimension=len(embeddings[0]) if embeddings else 0,
            processing_time=0.0
        )
        
        return response
        
    except Exception as e:
        logger.error(f"청크 임베딩 실패: {e}")
        raise HTTPException(status_code=500, detail=f"청크 임베딩 실패: {e}")


@router.post("/index", response_model=IndexingResponse)
async def index_chunks(
    request: IndexingRequest,
    background_tasks: BackgroundTasks,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """청크 인덱싱 (임베딩 + 인덱싱)"""
    try:
        logger.info(f"청크 인덱싱 요청: {len(request.chunks)}개 청크")
        
        # 1. 임베딩 생성
        embedded_chunks = await embedding_manager.embed_chunks(request.chunks)
        
        # 2. 벡터 인덱싱
        success = await vector_indexer.index_chunks(
            embedded_chunks, 
            request.collection_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="인덱싱 실패")
        
        response = IndexingResponse(
            success=True,
            message="청크 인덱싱이 완료되었습니다",
            indexed_count=len(embedded_chunks),
            collection_name=request.collection_name or "auto-generated",
            processing_time=0.0
        )
        
        return response
        
    except Exception as e:
        logger.error(f"청크 인덱싱 실패: {e}")
        raise HTTPException(status_code=500, detail=f"인덱싱 실패: {e}")


@router.post("/index-batch")
async def index_chunks_batch(
    request: IndexingRequest,
    background_tasks: BackgroundTasks,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """배치 청크 인덱싱"""
    try:
        logger.info(f"배치 청크 인덱싱 요청: {len(request.chunks)}개 청크")
        
        # 배치 처리기 생성
        batch_processor = BatchVectorizationProcessor(embedding_manager)
        
        # 배치 벡터화
        embedded_chunks = await batch_processor.process_chunks_batch(request.chunks)
        
        # 인덱싱
        success = await vector_indexer.index_chunks(
            embedded_chunks, 
            request.collection_name
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="배치 인덱싱 실패")
        
        # 배치 통계
        batch_stats = batch_processor.get_batch_stats()
        
        return {
            "success": True,
            "message": "배치 인덱싱이 완료되었습니다",
            "indexed_count": len(embedded_chunks),
            "collection_name": request.collection_name or "auto-generated",
            "batch_stats": batch_stats
        }
        
    except Exception as e:
        logger.error(f"배치 인덱싱 실패: {e}")
        raise HTTPException(status_code=500, detail=f"배치 인덱싱 실패: {e}")


@router.post("/search", response_model=SearchResponse)
async def search_similar_chunks(
    request: SearchRequest,
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """유사한 청크 검색"""
    try:
        logger.info(f"유사 청크 검색 요청: {request.query}")
        
        # 쿼리 임베딩 생성
        query_embedding = await embedding_manager.get_embedding_for_query(request.query)
        
        # 유사한 청크 검색
        results = await vector_indexer.search_similar_chunks(
            query_embedding,
            request.collection_name,
            request.top_k,
            request.filters
        )
        
        # 결과 포맷팅
        search_results = []
        for chunk, score in results:
            search_results.append({
                "chunk": chunk,
                "similarity_score": score
            })
        
        response = SearchResponse(
            success=True,
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            collection_name=request.collection_name or "all"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"유사 청크 검색 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검색 실패: {e}")


@router.get("/collections")
async def get_collections(
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """컬렉션 목록 조회"""
    try:
        collection_names = await vector_indexer.get_collection_names()
        
        # 각 컬렉션 정보 조회
        collections_info = []
        for name in collection_names:
            info = await vector_indexer.get_collection_info(name)
            collections_info.append(info)
        
        return {
            "success": True,
            "collections": collections_info,
            "total_collections": len(collections_info),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"컬렉션 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"컬렉션 조회 실패: {e}")


@router.get("/collections/{collection_name}")
async def get_collection_info(
    collection_name: str,
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """특정 컬렉션 정보 조회"""
    try:
        info = await vector_indexer.get_collection_info(collection_name)
        
        if not info:
            raise HTTPException(status_code=404, detail=f"컬렉션 '{collection_name}'을 찾을 수 없습니다")
        
        return {
            "success": True,
            "collection_info": info,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"컬렉션 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"컬렉션 정보 조회 실패: {e}")


@router.post("/verify", response_model=VerificationResponse)
async def verify_index_integrity(
    request: VerificationRequest,
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """인덱스 무결성 검증"""
    try:
        logger.info(f"인덱스 무결성 검증 요청: {request.collection_name}")
        
        # 무결성 검증 수행
        verification_result = await vector_indexer.verify_index_integrity(
            request.collection_name,
            request.sample_size
        )
        
        response = VerificationResponse(
            success=verification_result["valid"],
            collection_name=request.collection_name,
            verification_result=verification_result,
            message="인덱스 무결성 검증이 완료되었습니다" if verification_result["valid"] else "인덱스 무결성 검증 실패"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"인덱스 무결성 검증 실패: {e}")
        raise HTTPException(status_code=500, detail=f"무결성 검증 실패: {e}")


@router.delete("/collections/{collection_name}")
async def delete_collection(
    collection_name: str,
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """컬렉션 삭제"""
    try:
        success = await vector_indexer.delete_collection(collection_name)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"컬렉션 '{collection_name}'을 찾을 수 없습니다")
        
        return {
            "success": True,
            "message": f"컬렉션 '{collection_name}'이 삭제되었습니다",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"컬렉션 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"컬렉션 삭제 실패: {e}")


@router.get("/stats")
async def get_vectorization_stats(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """벡터화 통계 조회"""
    try:
        embedding_stats = embedding_manager.get_embedding_stats()
        indexing_stats = vector_indexer.get_indexing_stats()
        
        return {
            "success": True,
            "embedding_stats": embedding_stats,
            "indexing_stats": indexing_stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 조회 실패: {e}")


@router.post("/reset-stats")
async def reset_vectorization_stats(
    embedding_manager: EmbeddingManager = Depends(get_embedding_manager),
    vector_indexer: VectorIndexer = Depends(get_vector_indexer)
):
    """벡터화 통계 초기화"""
    try:
        embedding_manager.reset_stats()
        vector_indexer.reset_stats()
        
        return {
            "success": True,
            "message": "벡터화 통계가 초기화되었습니다",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"통계 초기화 실패: {e}")
        raise HTTPException(status_code=500, detail=f"통계 초기화 실패: {e}")