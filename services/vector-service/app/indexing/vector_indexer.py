"""
벡터 인덱서 - ChromaDB 및 FAISS 지원
"""
import asyncio
import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from shared.models.config import VectorDBConfig
from shared.models.document import Chunk, DocumentType
from shared.models.exceptions import VectorDBError
from shared.utils.logging import setup_logging, log_with_context

from app.indexing.chroma_client import ChromaDBClient
from app.indexing.faiss_client import FAISSClient

logger = setup_logging("vector-indexer")


class VectorIndexer:
    """벡터 인덱싱 관리 클래스"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.primary_client = None
        self.fallback_client = None
        self.is_initialized = False
        
        # 인덱싱 통계
        self.indexing_stats = {
            "total_vectors_indexed": 0,
            "successful_indexing": 0,
            "failed_indexing": 0,
            "total_indexing_time": 0.0,
            "avg_indexing_time": 0.0,
            "collections": {},
            "client_usage": {
                "chroma": 0,
                "faiss": 0
            }
        }
    
    async def initialize(self):
        """벡터 인덱서 초기화"""
        try:
            logger.info("벡터 인덱서 초기화 시작")
            
            # Primary 클라이언트 초기화 (ChromaDB)
            if self.config.provider == "chroma":
                try:
                    self.primary_client = ChromaDBClient(self.config)
                    await self.primary_client.initialize()
                    logger.info("ChromaDB 클라이언트 초기화 완료")
                except Exception as e:
                    logger.warning(f"ChromaDB 초기화 실패: {e}")
                    self.primary_client = None
            
            # Fallback 클라이언트 초기화 (FAISS)
            try:
                self.fallback_client = FAISSClient(self.config)
                await self.fallback_client.initialize()
                logger.info("FAISS 클라이언트 초기화 완료")
            except Exception as e:
                logger.warning(f"FAISS 초기화 실패: {e}")
                self.fallback_client = None
            
            # 최소 하나의 클라이언트는 사용 가능해야 함
            if not self.primary_client and not self.fallback_client:
                raise VectorDBError("모든 벡터 데이터베이스 클라이언트 초기화 실패")
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "벡터 인덱서 초기화 완료",
                primary_client_available=self.primary_client is not None,
                fallback_client_available=self.fallback_client is not None,
                config=self.config.__dict__
            )
            
        except Exception as e:
            logger.error(f"벡터 인덱서 초기화 실패: {e}")
            raise VectorDBError(f"초기화 실패: {e}")
    
    async def index_chunks(
        self, 
        chunks: List[Chunk], 
        collection_name: Optional[str] = None,
        use_fallback: bool = False
    ) -> bool:
        """청크 리스트를 벡터 데이터베이스에 인덱싱"""
        if not self.is_initialized:
            raise VectorDBError("벡터 인덱서가 초기화되지 않았습니다")
        
        if not chunks:
            return True
        
        start_time = time.time()
        
        try:
            # 컬렉션 이름 결정
            if not collection_name:
                collection_name = self._determine_collection_name(chunks[0])
            
            # 클라이언트 선택
            client = self._select_client(use_fallback)
            client_type = "faiss" if use_fallback else "chroma"
            
            log_with_context(
                logger, "info",
                "청크 인덱싱 시작",
                chunk_count=len(chunks),
                collection_name=collection_name,
                client_type=client_type
            )
            
            # 벡터가 없는 청크 필터링
            valid_chunks = [chunk for chunk in chunks if chunk.vector]
            if len(valid_chunks) != len(chunks):
                logger.warning(f"벡터가 없는 청크 {len(chunks) - len(valid_chunks)}개 제외")
            
            if not valid_chunks:
                logger.warning("인덱싱할 유효한 청크가 없습니다")
                return False
            
            # 인덱싱 수행
            success = await client.add_chunks(valid_chunks, collection_name)
            
            if success:
                # 통계 업데이트
                processing_time = time.time() - start_time
                self._update_indexing_stats(
                    len(valid_chunks), processing_time, collection_name, client_type, True
                )
                
                log_with_context(
                    logger, "info",
                    "청크 인덱싱 완료",
                    chunk_count=len(valid_chunks),
                    collection_name=collection_name,
                    processing_time=processing_time
                )
                
                return True
            else:
                raise VectorDBError("인덱싱 실패")
                
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_indexing_stats(
                len(chunks), processing_time, collection_name or "unknown", "unknown", False
            )
            
            # Primary 클라이언트 실패 시 Fallback 시도
            if not use_fallback and self.fallback_client:
                logger.warning(f"Primary 클라이언트 실패, Fallback 시도: {e}")
                return await self.index_chunks(chunks, collection_name, use_fallback=True)
            
            logger.error(f"청크 인덱싱 실패: {e}")
            raise VectorDBError(f"인덱싱 실패: {e}")
    
    async def search_similar_chunks(
        self, 
        query_vector: List[float], 
        collection_name: Optional[str] = None,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        use_fallback: bool = False
    ) -> List[Tuple[Chunk, float]]:
        """유사한 청크 검색"""
        if not self.is_initialized:
            raise VectorDBError("벡터 인덱서가 초기화되지 않았습니다")
        
        try:
            # 클라이언트 선택
            client = self._select_client(use_fallback)
            
            # 컬렉션 이름이 없으면 모든 컬렉션에서 검색
            if not collection_name:
                collection_names = await self.get_collection_names()
                if not collection_names:
                    return []
                collection_name = collection_names[0]  # 첫 번째 컬렉션 사용
            
            log_with_context(
                logger, "debug",
                "유사 청크 검색 시작",
                collection_name=collection_name,
                top_k=top_k,
                has_filters=filters is not None
            )
            
            # 검색 수행
            results = await client.search_similar(
                query_vector, collection_name, top_k, filters
            )
            
            # 임계값 필터링
            filtered_results = [
                (chunk, score) for chunk, score in results 
                if score >= self.config.similarity_threshold
            ]
            
            log_with_context(
                logger, "debug",
                "유사 청크 검색 완료",
                total_results=len(results),
                filtered_results=len(filtered_results),
                similarity_threshold=self.config.similarity_threshold
            )
            
            return filtered_results
            
        except Exception as e:
            # Primary 클라이언트 실패 시 Fallback 시도
            if not use_fallback and self.fallback_client:
                logger.warning(f"Primary 클라이언트 검색 실패, Fallback 시도: {e}")
                return await self.search_similar_chunks(
                    query_vector, collection_name, top_k, filters, use_fallback=True
                )
            
            logger.error(f"유사 청크 검색 실패: {e}")
            raise VectorDBError(f"검색 실패: {e}")
    
    async def get_collection_names(self, use_fallback: bool = False) -> List[str]:
        """컬렉션 목록 조회"""
        try:
            client = self._select_client(use_fallback)
            return await client.get_collection_names()
            
        except Exception as e:
            if not use_fallback and self.fallback_client:
                return await self.get_collection_names(use_fallback=True)
            
            logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []
    
    async def get_collection_info(
        self, 
        collection_name: str, 
        use_fallback: bool = False
    ) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        try:
            client = self._select_client(use_fallback)
            return await client.get_collection_info(collection_name)
            
        except Exception as e:
            if not use_fallback and self.fallback_client:
                return await self.get_collection_info(collection_name, use_fallback=True)
            
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    async def delete_collection(
        self, 
        collection_name: str, 
        use_fallback: bool = False
    ) -> bool:
        """컬렉션 삭제"""
        try:
            client = self._select_client(use_fallback)
            success = await client.delete_collection(collection_name)
            
            if success and collection_name in self.indexing_stats["collections"]:
                del self.indexing_stats["collections"][collection_name]
            
            return success
            
        except Exception as e:
            if not use_fallback and self.fallback_client:
                return await self.delete_collection(collection_name, use_fallback=True)
            
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
    
    async def verify_index_integrity(
        self, 
        collection_name: str, 
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """인덱스 무결성 검증"""
        try:
            # 컬렉션 정보 조회
            collection_info = await self.get_collection_info(collection_name)
            
            if not collection_info:
                return {"valid": False, "error": "컬렉션을 찾을 수 없습니다"}
            
            # 샘플 검색 테스트
            sample_results = []
            test_vector = [0.1] * 768  # 테스트용 더미 벡터
            
            start_time = time.time()
            search_results = await self.search_similar_chunks(
                test_vector, collection_name, top_k=sample_size
            )
            search_time = time.time() - start_time
            
            # 검증 결과
            verification_result = {
                "valid": True,
                "collection_name": collection_name,
                "collection_info": collection_info,
                "sample_search_results": len(search_results),
                "search_response_time": search_time,
                "meets_performance_threshold": search_time < 1.0,  # 1초 이내
                "timestamp": datetime.now().isoformat()
            }
            
            log_with_context(
                logger, "info",
                "인덱스 무결성 검증 완료",
                collection_name=collection_name,
                is_valid=verification_result["valid"],
                search_time=search_time
            )
            
            return verification_result
            
        except Exception as e:
            logger.error(f"인덱스 무결성 검증 실패: {e}")
            return {
                "valid": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _determine_collection_name(self, chunk: Chunk) -> str:
        """청크 기반 컬렉션 이름 결정"""
        if chunk.metadata and "document_type" in chunk.metadata:
            doc_type = chunk.metadata["document_type"]
        else:
            doc_type = "general"
        
        # 문서 타입별 컬렉션 이름
        collection_name = f"{self.config.collection_prefix}_{doc_type}"
        return collection_name.lower().replace(" ", "_")
    
    def _select_client(self, use_fallback: bool = False):
        """사용할 클라이언트 선택"""
        if use_fallback and self.fallback_client:
            return self.fallback_client
        elif self.primary_client:
            return self.primary_client
        elif self.fallback_client:
            return self.fallback_client
        else:
            raise VectorDBError("사용 가능한 벡터 데이터베이스 클라이언트가 없습니다")
    
    def _update_indexing_stats(
        self, 
        vector_count: int, 
        processing_time: float, 
        collection_name: str,
        client_type: str,
        success: bool
    ):
        """인덱싱 통계 업데이트"""
        self.indexing_stats["total_vectors_indexed"] += vector_count
        self.indexing_stats["total_indexing_time"] += processing_time
        
        if success:
            self.indexing_stats["successful_indexing"] += vector_count
            if client_type in self.indexing_stats["client_usage"]:
                self.indexing_stats["client_usage"][client_type] += vector_count
        else:
            self.indexing_stats["failed_indexing"] += vector_count
        
        # 컬렉션별 통계
        if collection_name not in self.indexing_stats["collections"]:
            self.indexing_stats["collections"][collection_name] = {
                "vector_count": 0,
                "last_updated": None
            }
        
        if success:
            self.indexing_stats["collections"][collection_name]["vector_count"] += vector_count
            self.indexing_stats["collections"][collection_name]["last_updated"] = datetime.now().isoformat()
        
        # 평균 인덱싱 시간 계산
        if self.indexing_stats["successful_indexing"] > 0:
            self.indexing_stats["avg_indexing_time"] = (
                self.indexing_stats["total_indexing_time"] / 
                self.indexing_stats["successful_indexing"]
            )
    
    def get_indexing_stats(self) -> Dict[str, Any]:
        """인덱싱 통계 조회"""
        return self.indexing_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.indexing_stats = {
            "total_vectors_indexed": 0,
            "successful_indexing": 0,
            "failed_indexing": 0,
            "total_indexing_time": 0.0,
            "avg_indexing_time": 0.0,
            "collections": {},
            "client_usage": {
                "chroma": 0,
                "faiss": 0
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.primary_client:
                await self.primary_client.cleanup()
            
            if self.fallback_client:
                await self.fallback_client.cleanup()
            
            self.is_initialized = False
            logger.info("벡터 인덱서 정리 완료")
            
        except Exception as e:
            logger.error(f"벡터 인덱서 정리 실패: {e}")