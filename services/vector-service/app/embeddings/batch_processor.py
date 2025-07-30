"""
배치 벡터화 처리기
"""
import asyncio
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from shared.models.document import Chunk
from shared.models.exceptions import VectorizationError
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("batch-processor")


class BatchVectorizationProcessor:
    """배치 벡터화 처리 클래스"""
    
    def __init__(self, embedding_manager, max_workers: int = 3):
        self.embedding_manager = embedding_manager
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 배치 처리 통계
        self.batch_stats = {
            "total_batches_processed": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_chunks_processed": 0,
            "total_processing_time": 0.0,
            "avg_batch_processing_time": 0.0,
            "current_batch_progress": 0.0,
            "is_processing": False
        }
        
        # 진행 상황 콜백
        self.progress_callbacks: List[Callable] = []
    
    async def process_chunks_batch(
        self, 
        chunks: List[Chunk],
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None
    ) -> List[Chunk]:
        """청크 배치 벡터화 처리"""
        if not chunks:
            return []
        
        batch_size = batch_size or self.embedding_manager.config.batch_size
        start_time = time.time()
        
        try:
            self.batch_stats["is_processing"] = True
            
            if progress_callback:
                self.progress_callbacks.append(progress_callback)
            
            log_with_context(
                logger, "info",
                "배치 벡터화 시작",
                total_chunks=len(chunks),
                batch_size=batch_size,
                estimated_batches=len(chunks) // batch_size + 1
            )
            
            # 청크를 배치로 분할
            chunk_batches = self._split_into_batches(chunks, batch_size)
            
            # 배치별 병렬 처리
            processed_chunks = await self._process_batches_parallel(chunk_batches)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_batch_stats(len(chunk_batches), len(chunks), processing_time, True)
            
            log_with_context(
                logger, "info",
                "배치 벡터화 완료",
                total_chunks=len(processed_chunks),
                total_batches=len(chunk_batches),
                processing_time=processing_time
            )
            
            return processed_chunks
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_batch_stats(0, len(chunks), processing_time, False)
            
            logger.error(f"배치 벡터화 실패: {e}")
            raise VectorizationError(f"배치 처리 실패: {e}")
            
        finally:
            self.batch_stats["is_processing"] = False
            self.batch_stats["current_batch_progress"] = 0.0
            if progress_callback in self.progress_callbacks:
                self.progress_callbacks.remove(progress_callback)
    
    def _split_into_batches(self, chunks: List[Chunk], batch_size: int) -> List[List[Chunk]]:
        """청크를 배치로 분할"""
        batches = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batches.append(batch)
        return batches
    
    async def _process_batches_parallel(self, chunk_batches: List[List[Chunk]]) -> List[Chunk]:
        """배치들을 병렬로 처리"""
        all_processed_chunks = []
        
        # 세마포어로 동시 처리 수 제한
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single_batch(batch_index: int, batch: List[Chunk]):
            async with semaphore:
                return await self._process_single_batch(batch_index, batch)
        
        # 모든 배치를 동시에 처리
        tasks = [
            process_single_batch(i, batch) 
            for i, batch in enumerate(chunk_batches)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 결과 수집 및 예외 처리
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"배치 {i} 처리 실패: {result}")
                # 실패한 배치는 원본 청크 반환 (벡터 없음)
                all_processed_chunks.extend(chunk_batches[i])
            else:
                all_processed_chunks.extend(result)
        
        return all_processed_chunks
    
    async def _process_single_batch(self, batch_index: int, batch: List[Chunk]) -> List[Chunk]:
        """단일 배치 처리"""
        batch_start_time = time.time()
        
        try:
            log_with_context(
                logger, "debug",
                "배치 처리 시작",
                batch_index=batch_index,
                batch_size=len(batch)
            )
            
            # 임베딩 생성
            processed_chunks = await self.embedding_manager.embed_chunks(batch)
            
            # 진행률 업데이트
            await self._update_progress(batch_index + 1)
            
            batch_time = time.time() - batch_start_time
            
            log_with_context(
                logger, "debug",
                "배치 처리 완료",
                batch_index=batch_index,
                batch_size=len(processed_chunks),
                processing_time=batch_time
            )
            
            return processed_chunks
            
        except Exception as e:
            batch_time = time.time() - batch_start_time
            
            log_with_context(
                logger, "error",
                "배치 처리 실패",
                batch_index=batch_index,
                batch_size=len(batch),
                processing_time=batch_time,
                error=str(e)
            )
            
            # 실패한 경우 원본 청크 반환
            return batch
    
    async def _update_progress(self, completed_batches: int):
        """진행률 업데이트"""
        if hasattr(self, '_total_batches'):
            progress = (completed_batches / self._total_batches) * 100
            self.batch_stats["current_batch_progress"] = progress
            
            # 콜백 호출
            for callback in self.progress_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(progress)
                    else:
                        callback(progress)
                except Exception as e:
                    logger.warning(f"진행률 콜백 실행 실패: {e}")
    
    async def process_chunks_with_retry(
        self, 
        chunks: List[Chunk],
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> List[Chunk]:
        """재시도 로직이 포함된 배치 처리"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                log_with_context(
                    logger, "info",
                    "배치 처리 시도",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    chunk_count=len(chunks)
                )
                
                result = await self.process_chunks_batch(chunks)
                
                # 성공한 청크 수 확인
                successful_chunks = [chunk for chunk in result if chunk.vector]
                success_rate = len(successful_chunks) / len(chunks) if chunks else 0
                
                if success_rate >= 0.8:  # 80% 이상 성공
                    logger.info(f"배치 처리 성공 (성공률: {success_rate:.1%})")
                    return result
                else:
                    logger.warning(f"배치 처리 성공률 낮음: {success_rate:.1%}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue
                    else:
                        return result
                        
            except Exception as e:
                last_exception = e
                logger.warning(f"배치 처리 시도 {attempt + 1} 실패: {e}")
                
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                else:
                    break
        
        # 모든 재시도 실패
        logger.error(f"배치 처리 최종 실패: {last_exception}")
        raise VectorizationError(f"배치 처리 실패 (재시도 {max_retries}회): {last_exception}")
    
    async def process_chunks_incremental(
        self, 
        chunks: List[Chunk],
        checkpoint_callback: Optional[Callable] = None,
        checkpoint_interval: int = 10
    ) -> List[Chunk]:
        """증분 처리 (체크포인트 지원)"""
        processed_chunks = []
        batch_size = self.embedding_manager.config.batch_size
        
        try:
            chunk_batches = self._split_into_batches(chunks, batch_size)
            self._total_batches = len(chunk_batches)
            
            for i, batch in enumerate(chunk_batches):
                try:
                    # 배치 처리
                    batch_result = await self._process_single_batch(i, batch)
                    processed_chunks.extend(batch_result)
                    
                    # 체크포인트 저장
                    if checkpoint_callback and (i + 1) % checkpoint_interval == 0:
                        await checkpoint_callback(processed_chunks, i + 1, len(chunk_batches))
                    
                except Exception as e:
                    logger.error(f"배치 {i} 처리 실패, 건너뛰기: {e}")
                    # 실패한 배치는 원본으로 추가
                    processed_chunks.extend(batch)
                    continue
            
            # 최종 체크포인트
            if checkpoint_callback:
                await checkpoint_callback(processed_chunks, len(chunk_batches), len(chunk_batches))
            
            return processed_chunks
            
        except Exception as e:
            logger.error(f"증분 처리 실패: {e}")
            raise VectorizationError(f"증분 처리 실패: {e}")
    
    def _update_batch_stats(
        self, 
        batch_count: int, 
        chunk_count: int, 
        processing_time: float, 
        success: bool
    ):
        """배치 통계 업데이트"""
        self.batch_stats["total_batches_processed"] += batch_count
        self.batch_stats["total_chunks_processed"] += chunk_count
        self.batch_stats["total_processing_time"] += processing_time
        
        if success:
            self.batch_stats["successful_batches"] += batch_count
        else:
            self.batch_stats["failed_batches"] += batch_count
        
        # 평균 처리 시간 계산
        if self.batch_stats["successful_batches"] > 0:
            self.batch_stats["avg_batch_processing_time"] = (
                self.batch_stats["total_processing_time"] / 
                self.batch_stats["successful_batches"]
            )
    
    def get_batch_stats(self) -> Dict[str, Any]:
        """배치 처리 통계 조회"""
        return self.batch_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.batch_stats = {
            "total_batches_processed": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_chunks_processed": 0,
            "total_processing_time": 0.0,
            "avg_batch_processing_time": 0.0,
            "current_batch_progress": 0.0,
            "is_processing": False
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            self.executor.shutdown(wait=True)
            self.progress_callbacks.clear()
            logger.info("배치 처리기 정리 완료")
            
        except Exception as e:
            logger.error(f"배치 처리기 정리 실패: {e}")