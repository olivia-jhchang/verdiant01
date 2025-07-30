"""
FAISS 클라이언트
"""
import asyncio
import json
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple

from shared.models.config import VectorDBConfig
from shared.models.document import Chunk
from shared.models.exceptions import VectorDBError
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("faiss-client")


class FAISSClient:
    """FAISS 클라이언트"""
    
    def __init__(self, config: VectorDBConfig):
        self.config = config
        self.indices = {}  # 컬렉션별 인덱스
        self.metadata_store = {}  # 메타데이터 저장소
        self.is_initialized = False
        self.storage_path = "data/faiss_indices"
    
    async def initialize(self):
        """FAISS 클라이언트 초기화"""
        try:
            # 저장 디렉토리 생성
            os.makedirs(self.storage_path, exist_ok=True)
            
            # 기존 인덱스 로드
            await self._load_existing_indices()
            
            self.is_initialized = True
            logger.info("FAISS 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"FAISS 클라이언트 초기화 실패: {e}")
            raise VectorDBError(f"FAISS 초기화 실패: {e}")
    
    async def add_chunks(self, chunks: List[Chunk], collection_name: str) -> bool:
        """청크를 인덱스에 추가"""
        if not self.is_initialized:
            raise VectorDBError("FAISS 클라이언트가 초기화되지 않았습니다")
        
        try:
            # 더미 FAISS 구현 (로컬 테스트용)
            if collection_name not in self.indices:
                self.indices[collection_name] = DummyFAISSIndex()
                self.metadata_store[collection_name] = {}
            
            index = self.indices[collection_name]
            metadata_store = self.metadata_store[collection_name]
            
            vectors = []
            for chunk in chunks:
                if chunk.vector:
                    vectors.append(chunk.vector)
                    metadata_store[len(metadata_store)] = {
                        "chunk": chunk,
                        "metadata": chunk.metadata or {}
                    }
            
            if vectors:
                await index.add_vectors(vectors)
                logger.info(f"컬렉션 {collection_name}에 {len(vectors)}개 벡터 추가")
            
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
        """유사한 청크 검색"""
        if not self.is_initialized:
            raise VectorDBError("FAISS 클라이언트가 초기화되지 않았습니다")
        
        try:
            if collection_name not in self.indices:
                return []
            
            index = self.indices[collection_name]
            metadata_store = self.metadata_store[collection_name]
            
            # 검색 수행
            indices, distances = await index.search(query_vector, top_k)
            
            results = []
            for idx, distance in zip(indices, distances):
                if idx in metadata_store:
                    chunk = metadata_store[idx]["chunk"]
                    # 거리를 유사도로 변환 (1 - normalized_distance)
                    similarity = max(0.0, 1.0 - distance)
                    results.append((chunk, similarity))
            
            logger.debug(f"컬렉션 {collection_name}에서 {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"유사 검색 실패: {e}")
            return []
    
    async def get_collection_names(self) -> List[str]:
        """컬렉션 목록 조회"""
        return list(self.indices.keys())
    
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """컬렉션 정보 조회"""
        if collection_name not in self.indices:
            return {}
        
        index = self.indices[collection_name]
        metadata_store = self.metadata_store[collection_name]
        
        return {
            "name": collection_name,
            "count": len(metadata_store),
            "dimension": index.dimension if hasattr(index, 'dimension') else 768,
            "index_type": "FAISS"
        }
    
    async def delete_collection(self, collection_name: str) -> bool:
        """컬렉션 삭제"""
        try:
            if collection_name in self.indices:
                del self.indices[collection_name]
            
            if collection_name in self.metadata_store:
                del self.metadata_store[collection_name]
            
            # 파일 시스템에서도 삭제
            index_file = os.path.join(self.storage_path, f"{collection_name}.index")
            metadata_file = os.path.join(self.storage_path, f"{collection_name}.metadata")
            
            for file_path in [index_file, metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            logger.info(f"컬렉션 {collection_name} 삭제")
            return True
            
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False
    
    async def _load_existing_indices(self):
        """기존 인덱스 로드"""
        try:
            if not os.path.exists(self.storage_path):
                return
            
            for filename in os.listdir(self.storage_path):
                if filename.endswith('.metadata'):
                    collection_name = filename[:-9]  # .metadata 제거
                    
                    # 메타데이터 로드
                    metadata_file = os.path.join(self.storage_path, filename)
                    with open(metadata_file, 'rb') as f:
                        self.metadata_store[collection_name] = pickle.load(f)
                    
                    # 더미 인덱스 생성
                    self.indices[collection_name] = DummyFAISSIndex()
                    
                    logger.info(f"컬렉션 {collection_name} 로드 완료")
            
        except Exception as e:
            logger.warning(f"기존 인덱스 로드 실패: {e}")
    
    async def _save_indices(self):
        """인덱스 저장"""
        try:
            for collection_name in self.metadata_store:
                metadata_file = os.path.join(self.storage_path, f"{collection_name}.metadata")
                with open(metadata_file, 'wb') as f:
                    pickle.dump(self.metadata_store[collection_name], f)
            
            logger.info("인덱스 저장 완료")
            
        except Exception as e:
            logger.error(f"인덱스 저장 실패: {e}")
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 인덱스 저장
            await self._save_indices()
            
            # 메모리 정리
            self.indices.clear()
            self.metadata_store.clear()
            
            self.is_initialized = False
            logger.info("FAISS 클라이언트 정리 완료")
            
        except Exception as e:
            logger.error(f"FAISS 클라이언트 정리 실패: {e}")


class DummyFAISSIndex:
    """더미 FAISS 인덱스 (로컬 테스트용)"""
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.vectors = []
        logger.info(f"더미 FAISS 인덱스 생성 (차원: {dimension})")
    
    async def add_vectors(self, vectors: List[List[float]]):
        """벡터 추가"""
        self.vectors.extend(vectors)
        logger.debug(f"{len(vectors)}개 벡터 추가, 총 {len(self.vectors)}개")
    
    async def search(self, query_vector: List[float], top_k: int) -> Tuple[List[int], List[float]]:
        """검색"""
        if not self.vectors:
            return [], []
        
        import numpy as np
        
        query_vec = np.array(query_vector)
        distances = []
        
        for i, stored_vector in enumerate(self.vectors):
            stored_vec = np.array(stored_vector)
            
            # 유클리드 거리 계산
            distance = np.linalg.norm(query_vec - stored_vec)
            distances.append((i, distance))
        
        # 거리 순으로 정렬
        distances.sort(key=lambda x: x[1])
        
        # Top-K 결과
        indices = [idx for idx, _ in distances[:top_k]]
        dists = [dist for _, dist in distances[:top_k]]
        
        return indices, dists