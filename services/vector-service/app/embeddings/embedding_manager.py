"""
임베딩 매니저 - BGE KoBase 모델 관리
"""
import asyncio
import time
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from shared.models.config import EmbeddingConfig
from shared.models.document import Chunk
from shared.models.exceptions import VectorizationError
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("embedding-manager")


class EmbeddingManager:
    """임베딩 생성 및 관리 클래스"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.primary_model = None
        self.fallback_model = None
        self.is_initialized = False
        
        # 임베딩 통계
        self.embedding_stats = {
            "total_embeddings_created": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "model_usage": {
                "primary": 0,
                "fallback": 0
            },
            "batch_stats": {
                "total_batches": 0,
                "avg_batch_size": 0.0,
                "avg_batch_time": 0.0
            }
        }
    
    async def initialize(self):
        """임베딩 매니저 초기화"""
        try:
            logger.info("임베딩 매니저 초기화 시작")
            
            # 모델 로딩을 별도 스레드에서 실행 (블로킹 방지)
            loop = asyncio.get_event_loop()
            
            # Primary 모델 로딩 (BGE KoBase)
            try:
                self.primary_model = await loop.run_in_executor(
                    None, self._load_primary_model
                )
                logger.info(f"Primary 모델 로딩 완료: {self.config.primary_model}")
            except Exception as e:
                logger.warning(f"Primary 모델 로딩 실패: {e}")
                self.primary_model = None
            
            # Fallback 모델 로딩
            try:
                self.fallback_model = await loop.run_in_executor(
                    None, self._load_fallback_model
                )
                logger.info(f"Fallback 모델 로딩 완료: {self.config.fallback_model}")
            except Exception as e:
                logger.warning(f"Fallback 모델 로딩 실패: {e}")
                self.fallback_model = None
            
            # 최소 하나의 모델은 로딩되어야 함
            if not self.primary_model and not self.fallback_model:
                raise VectorizationError("모든 임베딩 모델 로딩 실패")
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "임베딩 매니저 초기화 완료",
                primary_model_available=self.primary_model is not None,
                fallback_model_available=self.fallback_model is not None,
                config=self.config.__dict__
            )
            
        except Exception as e:
            logger.error(f"임베딩 매니저 초기화 실패: {e}")
            raise VectorizationError(f"초기화 실패: {e}")
    
    def _load_primary_model(self):
        """Primary 모델 로딩 (BGE KoBase)"""
        try:
            # 로컬 환경에서는 간단한 모델 사용
            if self.config.primary_model == "BAAI/bge-base-ko-v1.5":
                # 실제 환경에서는 sentence-transformers 사용
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer(self.config.primary_model, device=self.config.device)
                    return model
                except ImportError:
                    logger.warning("sentence-transformers 미설치, 더미 모델 사용")
                    return self._create_dummy_model("primary")
                except Exception as e:
                    logger.warning(f"BGE 모델 로딩 실패, 더미 모델 사용: {e}")
                    return self._create_dummy_model("primary")
            else:
                return self._create_dummy_model("primary")
                
        except Exception as e:
            logger.error(f"Primary 모델 로딩 실패: {e}")
            raise e
    
    def _load_fallback_model(self):
        """Fallback 모델 로딩"""
        try:
            # 로컬 환경에서는 더미 모델 사용
            return self._create_dummy_model("fallback")
                
        except Exception as e:
            logger.error(f"Fallback 모델 로딩 실패: {e}")
            raise e
    
    def _create_dummy_model(self, model_type: str):
        """더미 모델 생성 (로컬 테스트용)"""
        class DummyModel:
            def __init__(self, model_type, dimension=768):
                self.model_type = model_type
                self.dimension = dimension
                logger.info(f"더미 {model_type} 모델 생성 (차원: {dimension})")
            
            def encode(self, texts, batch_size=32, show_progress_bar=False, **kwargs):
                """더미 임베딩 생성"""
                if isinstance(texts, str):
                    texts = [texts]
                
                # 텍스트 기반 의사 임베딩 생성 (재현 가능)
                embeddings = []
                for text in texts:
                    # 텍스트 해시를 시드로 사용하여 일관된 벡터 생성
                    np.random.seed(hash(text) % 2**32)
                    embedding = np.random.normal(0, 1, self.dimension).astype(np.float32)
                    # 정규화
                    embedding = embedding / np.linalg.norm(embedding)
                    embeddings.append(embedding)
                
                return np.array(embeddings)
        
        return DummyModel(model_type, self.config.vector_dimension)
    
    async def embed_texts(
        self, 
        texts: List[str], 
        use_fallback: bool = False
    ) -> List[List[float]]:
        """텍스트 리스트를 임베딩으로 변환"""
        if not self.is_initialized:
            raise VectorizationError("임베딩 매니저가 초기화되지 않았습니다")
        
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # 모델 선택
            model = self._select_model(use_fallback)
            model_type = "fallback" if use_fallback else "primary"
            
            log_with_context(
                logger, "info",
                "텍스트 임베딩 시작",
                text_count=len(texts),
                model_type=model_type,
                batch_size=self.config.batch_size
            )
            
            # 배치 처리
            embeddings = await self._process_embeddings_in_batches(texts, model)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_embedding_stats(len(texts), processing_time, model_type, True)
            
            log_with_context(
                logger, "info",
                "텍스트 임베딩 완료",
                text_count=len(texts),
                embedding_dimension=len(embeddings[0]) if embeddings else 0,
                processing_time=processing_time
            )
            
            return embeddings
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_embedding_stats(len(texts), processing_time, "unknown", False)
            
            # Primary 모델 실패 시 Fallback 시도
            if not use_fallback and self.fallback_model:
                logger.warning(f"Primary 모델 실패, Fallback 모델 시도: {e}")
                return await self.embed_texts(texts, use_fallback=True)
            
            logger.error(f"텍스트 임베딩 실패: {e}")
            raise VectorizationError(f"임베딩 생성 실패: {e}")
    
    async def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """청크 리스트를 임베딩하여 벡터 추가"""
        if not chunks:
            return []
        
        try:
            # 청크에서 텍스트 추출
            texts = [chunk.text for chunk in chunks]
            
            # 임베딩 생성
            embeddings = await self.embed_texts(texts)
            
            # 청크에 벡터 추가
            for chunk, embedding in zip(chunks, embeddings):
                chunk.vector = embedding
                
                # 메타데이터에 임베딩 정보 추가
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.update({
                    "embedding_model": self.config.primary_model,
                    "embedding_dimension": len(embedding),
                    "embedding_created_at": datetime.now().isoformat()
                })
            
            log_with_context(
                logger, "info",
                "청크 임베딩 완료",
                chunk_count=len(chunks),
                embedding_dimension=len(embeddings[0]) if embeddings else 0
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"청크 임베딩 실패: {e}")
            raise VectorizationError(f"청크 임베딩 실패: {e}")
    
    async def _process_embeddings_in_batches(
        self, 
        texts: List[str], 
        model
    ) -> List[List[float]]:
        """배치 단위로 임베딩 처리"""
        all_embeddings = []
        batch_size = self.config.batch_size
        
        # 배치별 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_start_time = time.time()
            
            try:
                # 비동기 처리를 위해 executor 사용
                loop = asyncio.get_event_loop()
                batch_embeddings = await loop.run_in_executor(
                    None, 
                    lambda: model.encode(
                        batch_texts,
                        batch_size=len(batch_texts),
                        show_progress_bar=False
                    )
                )
                
                # numpy array를 list로 변환
                batch_embeddings_list = batch_embeddings.tolist()
                all_embeddings.extend(batch_embeddings_list)
                
                # 배치 통계 업데이트
                batch_time = time.time() - batch_start_time
                self._update_batch_stats(len(batch_texts), batch_time)
                
                log_with_context(
                    logger, "debug",
                    "배치 임베딩 완료",
                    batch_index=i // batch_size + 1,
                    batch_size=len(batch_texts),
                    batch_time=batch_time
                )
                
            except Exception as e:
                logger.error(f"배치 {i // batch_size + 1} 처리 실패: {e}")
                # 실패한 배치는 0 벡터로 대체
                zero_embeddings = [[0.0] * self.config.vector_dimension] * len(batch_texts)
                all_embeddings.extend(zero_embeddings)
        
        return all_embeddings
    
    def _select_model(self, use_fallback: bool = False):
        """사용할 모델 선택"""
        if use_fallback and self.fallback_model:
            return self.fallback_model
        elif self.primary_model:
            return self.primary_model
        elif self.fallback_model:
            return self.fallback_model
        else:
            raise VectorizationError("사용 가능한 임베딩 모델이 없습니다")
    
    def _update_embedding_stats(
        self, 
        text_count: int, 
        processing_time: float, 
        model_type: str, 
        success: bool
    ):
        """임베딩 통계 업데이트"""
        self.embedding_stats["total_embeddings_created"] += text_count
        self.embedding_stats["total_processing_time"] += processing_time
        
        if success:
            self.embedding_stats["successful_embeddings"] += text_count
            if model_type in self.embedding_stats["model_usage"]:
                self.embedding_stats["model_usage"][model_type] += text_count
        else:
            self.embedding_stats["failed_embeddings"] += text_count
        
        # 평균 처리 시간 계산
        if self.embedding_stats["successful_embeddings"] > 0:
            self.embedding_stats["avg_processing_time"] = (
                self.embedding_stats["total_processing_time"] / 
                self.embedding_stats["successful_embeddings"]
            )
    
    def _update_batch_stats(self, batch_size: int, batch_time: float):
        """배치 통계 업데이트"""
        batch_stats = self.embedding_stats["batch_stats"]
        batch_stats["total_batches"] += 1
        
        # 평균 배치 크기 계산
        total_batches = batch_stats["total_batches"]
        prev_avg_size = batch_stats["avg_batch_size"]
        batch_stats["avg_batch_size"] = (
            (prev_avg_size * (total_batches - 1) + batch_size) / total_batches
        )
        
        # 평균 배치 시간 계산
        prev_avg_time = batch_stats["avg_batch_time"]
        batch_stats["avg_batch_time"] = (
            (prev_avg_time * (total_batches - 1) + batch_time) / total_batches
        )
    
    async def get_embedding_for_query(self, query: str) -> List[float]:
        """검색 쿼리용 임베딩 생성"""
        try:
            embeddings = await self.embed_texts([query])
            return embeddings[0] if embeddings else []
            
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 실패: {e}")
            raise VectorizationError(f"쿼리 임베딩 실패: {e}")
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """임베딩 통계 조회"""
        return self.embedding_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.embedding_stats = {
            "total_embeddings_created": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "model_usage": {
                "primary": 0,
                "fallback": 0
            },
            "batch_stats": {
                "total_batches": 0,
                "avg_batch_size": 0.0,
                "avg_batch_time": 0.0
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 모델 메모리 해제
            self.primary_model = None
            self.fallback_model = None
            self.is_initialized = False
            
            logger.info("임베딩 매니저 정리 완료")
            
        except Exception as e:
            logger.error(f"임베딩 매니저 정리 실패: {e}")