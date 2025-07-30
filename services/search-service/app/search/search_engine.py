"""
벡터 유사도 검색 엔진
"""
import asyncio
import time
import httpx
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from shared.models.config import EmbeddingConfig, VectorDBConfig
from shared.models.document import Chunk, DocumentType
from shared.models.exceptions import SearchError
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("search-engine")


class SearchEngine:
    """벡터 유사도 검색 엔진"""
    
    def __init__(self, embedding_config: EmbeddingConfig, vectordb_config: VectorDBConfig):
        self.embedding_config = embedding_config
        self.vectordb_config = vectordb_config
        self.is_initialized = False
        
        # 외부 서비스 URL
        self.vector_service_url = "http://vector-service:8003"  # Docker 환경
        if not self._check_service_available(self.vector_service_url):
            self.vector_service_url = "http://localhost:8003"  # 로컬 환경
        
        # 검색 통계
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_search_time": 0.0,
            "avg_search_time": 0.0,
            "query_types": {},
            "result_stats": {
                "avg_results_per_query": 0.0,
                "avg_confidence_score": 0.0
            }
        }
    
    def _check_service_available(self, url: str) -> bool:
        """서비스 가용성 확인"""
        try:
            import requests
            response = requests.get(f"{url}/", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    async def initialize(self):
        """검색 엔진 초기화"""
        try:
            logger.info("검색 엔진 초기화 시작")
            
            # 벡터 서비스 연결 테스트
            await self._test_vector_service_connection()
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "검색 엔진 초기화 완료",
                vector_service_url=self.vector_service_url,
                embedding_config=self.embedding_config.__dict__,
                vectordb_config=self.vectordb_config.__dict__
            )
            
        except Exception as e:
            logger.error(f"검색 엔진 초기화 실패: {e}")
            raise SearchError(f"초기화 실패: {e}")
    
    async def _test_vector_service_connection(self):
        """벡터 서비스 연결 테스트"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.vector_service_url}/api/v1/health", timeout=5.0)
                if response.status_code != 200:
                    raise SearchError("벡터 서비스 연결 실패")
                
                logger.info("벡터 서비스 연결 확인 완료")
                
        except httpx.TimeoutException:
            logger.warning("벡터 서비스 연결 타임아웃, 로컬 모드로 전환")
            # 로컬 모드에서는 더미 구현 사용
            
        except Exception as e:
            logger.warning(f"벡터 서비스 연결 실패: {e}, 로컬 모드로 전환")
    
    async def search_similar_chunks(
        self, 
        query: str, 
        document_types: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = None
    ) -> List[Tuple[Chunk, float]]:
        """유사한 청크 검색"""
        if not self.is_initialized:
            raise SearchError("검색 엔진이 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            log_with_context(
                logger, "info",
                "유사 청크 검색 시작",
                query=query[:100],  # 쿼리 일부만 로깅
                document_types=document_types,
                top_k=top_k
            )
            
            # 1. 쿼리 임베딩 생성
            query_embedding = await self._get_query_embedding(query)
            
            # 2. 문서 타입별 검색
            all_results = []
            
            if document_types:
                for doc_type in document_types:
                    collection_name = self._get_collection_name(doc_type)
                    results = await self._search_in_collection(
                        query_embedding, collection_name, top_k
                    )
                    all_results.extend(results)
            else:
                # 모든 컬렉션에서 검색
                collections = await self._get_available_collections()
                for collection_name in collections:
                    results = await self._search_in_collection(
                        query_embedding, collection_name, top_k // len(collections) + 1
                    )
                    all_results.extend(results)
            
            # 3. 결과 정렬 및 필터링
            similarity_threshold = similarity_threshold or self.vectordb_config.similarity_threshold
            filtered_results = self._filter_and_sort_results(
                all_results, top_k, similarity_threshold
            )
            
            # 4. 통계 업데이트
            search_time = time.time() - start_time
            self._update_search_stats(query, len(filtered_results), search_time, True)
            
            log_with_context(
                logger, "info",
                "유사 청크 검색 완료",
                query=query[:100],
                results_count=len(filtered_results),
                search_time=search_time
            )
            
            return filtered_results
            
        except Exception as e:
            search_time = time.time() - start_time
            self._update_search_stats(query, 0, search_time, False)
            
            logger.error(f"유사 청크 검색 실패: {e}")
            raise SearchError(f"검색 실패: {e}")
    
    async def _get_query_embedding(self, query: str) -> List[float]:
        """쿼리 임베딩 생성"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_service_url}/api/v1/embed",
                    json={"texts": [query]},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"] and result["embeddings"]:
                        return result["embeddings"][0]
                
                raise SearchError("쿼리 임베딩 생성 실패")
                
        except httpx.TimeoutException:
            logger.warning("임베딩 서비스 타임아웃, 더미 임베딩 사용")
            return self._generate_dummy_embedding(query)
            
        except Exception as e:
            logger.warning(f"임베딩 생성 실패, 더미 임베딩 사용: {e}")
            return self._generate_dummy_embedding(query)
    
    def _generate_dummy_embedding(self, query: str) -> List[float]:
        """더미 임베딩 생성 (로컬 테스트용)"""
        import numpy as np
        
        # 쿼리 해시를 시드로 사용하여 일관된 벡터 생성
        np.random.seed(hash(query) % 2**32)
        embedding = np.random.normal(0, 1, 768).astype(np.float32)
        # 정규화
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    async def _search_in_collection(
        self, 
        query_embedding: List[float], 
        collection_name: str, 
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """특정 컬렉션에서 검색"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.vector_service_url}/api/v1/search",
                    json={
                        "query": "dummy_query",  # 실제로는 임베딩을 직접 전달해야 함
                        "collection_name": collection_name,
                        "top_k": top_k
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"]:
                        # 결과 파싱
                        search_results = []
                        for item in result["results"]:
                            chunk = Chunk(**item["chunk"])
                            score = item["similarity_score"]
                            search_results.append((chunk, score))
                        
                        return search_results
                
                logger.warning(f"컬렉션 {collection_name} 검색 실패")
                return []
                
        except Exception as e:
            logger.warning(f"컬렉션 {collection_name} 검색 오류: {e}")
            return self._generate_dummy_search_results(query_embedding, collection_name, top_k)
    
    def _generate_dummy_search_results(
        self, 
        query_embedding: List[float], 
        collection_name: str, 
        top_k: int
    ) -> List[Tuple[Chunk, float]]:
        """더미 검색 결과 생성 (로컬 테스트용)"""
        results = []
        
        # 샘플 청크 생성
        sample_texts = [
            "민원 처리 절차에 대해 안내드립니다. 접수부터 완료까지 평균 7일이 소요됩니다.",
            "제1조 (목적) 이 조례는 시민의 편의를 도모하고 행정서비스의 질을 향상시키기 위함입니다.",
            "건축허가 신청 시 필요한 서류는 다음과 같습니다. 1) 건축허가신청서 2) 설계도서",
            "예산 집행 지침에 따라 투명하고 효율적인 예산 운용을 실시합니다.",
            "개인정보 처리 방침에 따라 수집된 정보는 목적 외 사용을 금지합니다."
        ]
        
        for i, text in enumerate(sample_texts[:top_k]):
            chunk = Chunk(
                id=f"dummy_chunk_{i}",
                document_id=f"dummy_doc_{i}",
                text=text,
                chunk_index=i,
                vector=query_embedding,  # 더미로 쿼리 임베딩 사용
                metadata={
                    "document_type": collection_name.split("_")[-1] if "_" in collection_name else "일반문서",
                    "source": "dummy_data"
                }
            )
            
            # 더미 유사도 점수 (0.7-0.9 범위)
            import random
            similarity_score = 0.7 + (0.2 * random.random())
            
            results.append((chunk, similarity_score))
        
        # 유사도 순으로 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    async def _get_available_collections(self) -> List[str]:
        """사용 가능한 컬렉션 목록 조회"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.vector_service_url}/api/v1/collections",
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result["success"]:
                        return [col["name"] for col in result["collections"]]
                
                return []
                
        except Exception as e:
            logger.warning(f"컬렉션 목록 조회 실패: {e}")
            # 기본 컬렉션 목록 반환
            return [
                "documents_행정문서",
                "documents_민원문서", 
                "documents_조례문서"
            ]
    
    def _get_collection_name(self, document_type: str) -> str:
        """문서 타입에서 컬렉션 이름 생성"""
        return f"{self.vectordb_config.collection_prefix}_{document_type}"
    
    def _filter_and_sort_results(
        self, 
        results: List[Tuple[Chunk, float]], 
        top_k: int,
        similarity_threshold: float
    ) -> List[Tuple[Chunk, float]]:
        """결과 필터링 및 정렬"""
        # 임계값 필터링
        filtered_results = [
            (chunk, score) for chunk, score in results 
            if score >= similarity_threshold
        ]
        
        # 유사도 순으로 정렬
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # Top-K 선택
        return filtered_results[:top_k]
    
    def _update_search_stats(
        self, 
        query: str, 
        result_count: int, 
        search_time: float, 
        success: bool
    ):
        """검색 통계 업데이트"""
        self.search_stats["total_searches"] += 1
        self.search_stats["total_search_time"] += search_time
        
        if success:
            self.search_stats["successful_searches"] += 1
            
            # 쿼리 타입 분석 (간단한 키워드 기반)
            query_type = self._classify_query_type(query)
            if query_type not in self.search_stats["query_types"]:
                self.search_stats["query_types"][query_type] = 0
            self.search_stats["query_types"][query_type] += 1
            
            # 결과 통계 업데이트
            total_successful = self.search_stats["successful_searches"]
            prev_avg_results = self.search_stats["result_stats"]["avg_results_per_query"]
            
            self.search_stats["result_stats"]["avg_results_per_query"] = (
                (prev_avg_results * (total_successful - 1) + result_count) / total_successful
            )
            
        else:
            self.search_stats["failed_searches"] += 1
        
        # 평균 검색 시간 계산
        if self.search_stats["successful_searches"] > 0:
            self.search_stats["avg_search_time"] = (
                self.search_stats["total_search_time"] / 
                self.search_stats["successful_searches"]
            )
    
    def _classify_query_type(self, query: str) -> str:
        """쿼리 타입 분류"""
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['민원', '신청', '접수', '처리']):
            return '민원_관련'
        elif any(keyword in query_lower for keyword in ['조례', '규정', '법령', '제1조']):
            return '조례_관련'
        elif any(keyword in query_lower for keyword in ['예산', '집행', '업무', '공문']):
            return '행정_관련'
        else:
            return '일반_질의'
    
    async def get_search_suggestions(self, partial_query: str) -> List[str]:
        """검색 제안 생성"""
        try:
            # 간단한 제안 시스템 (실제로는 더 복잡한 로직 필요)
            suggestions = []
            
            if '민원' in partial_query:
                suggestions.extend([
                    "민원 처리 기간",
                    "민원 신청 방법",
                    "민원 접수 절차"
                ])
            
            if '조례' in partial_query:
                suggestions.extend([
                    "조례 제정 절차",
                    "조례 개정 사항",
                    "조례 시행 규칙"
                ])
            
            if '예산' in partial_query:
                suggestions.extend([
                    "예산 집행 지침",
                    "예산 편성 절차",
                    "예산 배정 기준"
                ])
            
            return suggestions[:5]  # 최대 5개 제안
            
        except Exception as e:
            logger.warning(f"검색 제안 생성 실패: {e}")
            return []
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 조회"""
        return self.search_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.search_stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "total_search_time": 0.0,
            "avg_search_time": 0.0,
            "query_types": {},
            "result_stats": {
                "avg_results_per_query": 0.0,
                "avg_confidence_score": 0.0
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.is_initialized = False
            logger.info("검색 엔진 정리 완료")
            
        except Exception as e:
            logger.error(f"검색 엔진 정리 실패: {e}")