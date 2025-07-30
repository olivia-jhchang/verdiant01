"""
ChromaDB 클라이언트
"""
import asyncio
import json
from typing import List, Dict, Any, Optional, Tuple

from shared.models.config import VectorDBConfig
from shared.models.document import Chunk
from shared.models.exceptions import VectorDBError
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("chroma-client")


class ChromaDBClient:
    """ChromaDB 클라이언트"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.client = None
        self.is_initialized = False
    
    async def initialize(self):
        """ChromaDB 클라이언트 초기화"""
        try:
            # ChromaDB 클라이언트 생성 (로컬 환경용 더미 구현)
            self.client = DummyChromaClient(self.config)
            await self.client.initialize()
            
            self.is_initialized = True
            logger.info("ChromaDB 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {e}")
            raise VectorDBError(f"ChromaDB 초기화 실패: {e}")
    
    async def add_chunks(self, chunks: List[Chunk], collection_name: str) -> bool:
        """청크를 컬렉션에 추가"""
        if not self.is_initialized:
            raise VectorDBError("ChromaDB 클라이언트가 초기화되지 않았습니다")
        
        try:
            return await self.client.add_chunks(chunks, collection_name)
            
        except Exception as e:
            logger.error(f"청크 추가 실패: {e}")
            raise VectorDBError(f"청크 추가 실패: {e}")
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        collection_name: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """유사한 청크 검색"""
        if not self.is_initialized:
            raise VectorDBError("ChromaDB 클라이언트가 초기화되지 않았습니다")
        
        try:
            return await self.client.search_similar(query_vector, collection_name, top_k, filters)
            
        except Exception as e:
            logger.error(f"유사 검색 실패: {e}")
            raise VectorDBError(f"검색 실패: {e}")
    
    async def get_collection_names(self) -> List[str]:
        """컬렉션 목록 조회"""
        if not self.is_initialized:
            return []
        
        try:
            return await self.client.get_collection_names()
            
        except Exception as e:
            logger.error(f"컬렉션 목록 조회 실패: {e}")
            return []
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        if not self.is_initialized:
            return {}
        
        try:
            return await self.client.get_collection_info(collection_name)
            
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}
    
    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        if not self.is_initialized:
            return False
        
        try:
            return await self.client.delete_collection(collection_name)
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            if self.client:
                await self.client.cleanup()
            
            self.is_initialized = False
            logger.info("ChromaDB 클라이언트 정리 완료")
            
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 정리 실패: {e}")


class DummyChromaClient:
    """더미 ChromaDB 클라이언트 (로컬 테스트용)"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.collections = {}  # 메모리 기반 저장소
        logger.info("더미 ChromaDB 클라이언트 생성")
    
    async def initialize(self):
        """초기화"""
        logger.info("더미 ChromaDB 클라이언트 초기화 완료")
    
    async def add_chunks(self, chunks: List[Chunk], collection_name: str) -> bool:
        """청크 추가"""
        try:
            if collection_name not in self.collections:
                self.collections[collection_name] = {
                    "chunks": [],
                    "vectors": [],
                    "metadatas": [],
                    "ids": []
                }
            
            collection = self.collections[collection_name]
            
            for chunk in chunks:
                if chunk.vector:
                    collection["chunks"].append(chunk)
                    collection["vectors"].append(chunk.vector)
                    collection["metadatas"].append(chunk.metadata or {})
                    collection["ids"].append(chunk.id)
            
            logger.info(f"컬렉션 {collection_name}에 {len(chunks)}개 청크 추가")
            return True
            
        except Exception as e:
            logger.error(f"청크 추가 실패: {e}")
            return False
    
    async def search_similar(
        self, 
        query_vector: List[float], 
        collection_name: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Chunk, float]]:
        """유사 검색"""
        try:
            if collection_name not in self.collections:
                return []
            
            collection = self.collections[collection_name]
            if not collection["vectors"]:
                return []
            
            # 코사인 유사도 계산
            import numpy as np
            
            query_vec = np.array(query_vector)
            similarities = []
            
            for i, stored_vector in enumerate(collection["vectors"]):
                stored_vec = np.array(stored_vector)
                
                # 코사인 유사도
                dot_product = np.dot(query_vec, stored_vec)
                norm_query = np.linalg.norm(query_vec)
                norm_stored = np.linalg.norm(stored_vec)
                
                if norm_query > 0 and norm_stored > 0:
                    similarity = dot_product / (norm_query * norm_stored)
                else:
                    similarity = 0.0
                
                similarities.append((i, similarity))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Top-K 결과 반환
            results = []
            for i, similarity in similarities[:top_k]:
                chunk = collection["chunks"][i]
                results.append((chunk, similarity))
            
            logger.debug(f"컬렉션 {collection_name}에서 {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"유사 검색 실패: {e}")
            return []
    
    async def get_collection_names(self) -> List[str]:
        """컬렉션 목록"""
        return list(self.collections.keys())
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보"""
        if collection_name not in self.collections:
            return {}
        
        collection = self.collections[collection_name]
        return {
            "name": collection_name,
            "count": len(collection["chunks"]),
            "dimension": len(collection["vectors"][0]) if collection["vectors"] else 0
        }
    
    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        if collection_name in self.collections:
            del self.collections[collection_name]
            logger.info(f"컬렉션 {collection_name} 삭제")
            return True
        return False
    
    async def cleanup(self):
        """정리"""
        self.collections.clear()
        logger.info("더미 ChromaDB 클라이언트 정리 완료")