"""
문서 처리 메인 프로세서
"""
import asyncio
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from shared.models.document import Document, Chunk, DocumentType
from shared.models.config import ChunkingConfig
from shared.models.exceptions import DocumentProcessingError, ChunkingError
from shared.utils.logging import setup_logging, log_with_context
from shared.utils.korean_utils import (
    clean_korean_text, 
    detect_document_structure,
    chunk_by_structure
)

from app.processors.structure_analyzer import StructureAnalyzer
from app.chunkers.korean_chunker import KoreanChunker
from app.classifiers.document_classifier import DocumentClassifier

logger = setup_logging("document-processor")


class DocumentProcessor:
    """문서 처리 메인 클래스"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.structure_analyzer = StructureAnalyzer()
        self.korean_chunker = KoreanChunker(config)
        self.document_classifier = DocumentClassifier()
        
        self.processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "total_chunks_created": 0,
            "processing_time_total": 0.0,
            "type_distribution": {
                "행정문서": 0,
                "민원문서": 0,
                "조례문서": 0
            }
        }
        
        self.is_initialized = False
    
    async def initialize(self):
        """프로세서 초기화"""
        try:
            # 구조 분석기 초기화
            await self.structure_analyzer.initialize()
            
            # 한국어 청킹 모델 초기화
            await self.korean_chunker.initialize()
            
            # 문서 분류기 초기화
            await self.document_classifier.initialize()
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "문서 처리기 초기화 완료",
                config=self.config.__dict__
            )
            
        except Exception as e:
            logger.error(f"문서 처리기 초기화 실패: {e}")
            raise DocumentProcessingError(f"초기화 실패: {e}")
    
    async def process_document(self, document: Document) -> List[Chunk]:
        """단일 문서 처리"""
        if not self.is_initialized:
            raise DocumentProcessingError("문서 처리기가 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            log_with_context(
                logger, "info",
                "문서 처리 시작",
                document_id=document.id,
                document_type=document.document_type.value,
                content_length=len(document.content)
            )
            
            # 1. 문서 구조 분석
            structure_info = await self._analyze_document_structure(document)
            
            # 2. 문서 분류 재검증
            refined_type = await self._refine_document_classification(document)
            if refined_type != document.document_type:
                document.document_type = refined_type
                log_with_context(
                    logger, "info",
                    "문서 타입 재분류",
                    document_id=document.id,
                    original_type=document.document_type.value,
                    refined_type=refined_type.value
                )
            
            # 3. 텍스트 정리 및 정규화
            cleaned_content = await self._clean_and_normalize_text(document.content)
            
            # 4. 청킹 수행
            chunks = await self._chunk_document(
                document, cleaned_content, structure_info
            )
            
            # 5. 청크 후처리
            processed_chunks = await self._post_process_chunks(chunks, document)
            
            # 통계 업데이트
            processing_time = time.time() - start_time
            self._update_processing_stats(document, len(processed_chunks), processing_time)
            
            log_with_context(
                logger, "info",
                "문서 처리 완료",
                document_id=document.id,
                chunks_created=len(processed_chunks),
                processing_time=processing_time
            )
            
            return processed_chunks
            
        except Exception as e:
            self.processing_stats["failed_processing"] += 1
            processing_time = time.time() - start_time
            
            log_with_context(
                logger, "error",
                "문서 처리 실패",
                document_id=document.id,
                error=str(e),
                processing_time=processing_time
            )
            
            raise DocumentProcessingError(f"문서 처리 실패 (ID: {document.id}): {e}")
    
    async def process_documents_batch(self, documents: List[Document]) -> List[List[Chunk]]:
        """문서 배치 처리"""
        if not documents:
            return []
        
        log_with_context(
            logger, "info",
            "배치 문서 처리 시작",
            document_count=len(documents)
        )
        
        results = []
        
        # 동시 처리 (최대 5개씩)
        semaphore = asyncio.Semaphore(5)
        
        async def process_single(doc):
            async with semaphore:
                return await self.process_document(doc)
        
        try:
            # 모든 문서를 동시에 처리
            tasks = [process_single(doc) for doc in documents]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"문서 {documents[i].id} 처리 실패: {result}")
                    successful_results.append([])  # 빈 청크 리스트
                else:
                    successful_results.append(result)
            
            log_with_context(
                logger, "info",
                "배치 문서 처리 완료",
                total_documents=len(documents),
                successful_documents=len([r for r in results if not isinstance(r, Exception)])
            )
            
            return successful_results
            
        except Exception as e:
            logger.error(f"배치 처리 실패: {e}")
            raise DocumentProcessingError(f"배치 처리 실패: {e}")
    
    async def _analyze_document_structure(self, document: Document) -> Dict[str, Any]:
        """문서 구조 분석"""
        try:
            # 기본 구조 감지
            basic_structure = detect_document_structure(document.content)
            
            # 상세 구조 분석
            detailed_structure = await self.structure_analyzer.analyze_structure(
                document.content, document.document_type
            )
            
            # 구조 정보 통합
            structure_info = {
                **basic_structure,
                **detailed_structure,
                "analysis_time": datetime.now().isoformat()
            }
            
            return structure_info
            
        except Exception as e:
            logger.warning(f"구조 분석 실패: {e}")
            return {"document_type": "일반문서", "has_structure": False}
    
    async def _refine_document_classification(self, document: Document) -> DocumentType:
        """문서 분류 재검증"""
        try:
            # 내용 기반 분류 수행
            predicted_type = await self.document_classifier.classify_document(
                document.title, document.content
            )
            
            # 기존 분류와 비교
            if predicted_type != document.document_type:
                confidence = await self.document_classifier.get_classification_confidence(
                    document.title, document.content, predicted_type
                )
                
                # 높은 신뢰도인 경우에만 변경
                if confidence > 0.8:
                    return predicted_type
            
            return document.document_type
            
        except Exception as e:
            logger.warning(f"문서 분류 재검증 실패: {e}")
            return document.document_type
    
    async def _clean_and_normalize_text(self, content: str) -> str:
        """텍스트 정리 및 정규화"""
        try:
            # 기본 정리
            cleaned = clean_korean_text(content)
            
            # 추가 정규화 (문서 처리 서비스 특화)
            # 1. 연속된 공백 정리
            import re
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
            # 2. 불필요한 특수문자 제거
            cleaned = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}""''「」『』\-]', '', cleaned)
            
            # 3. 문장 경계 정리
            cleaned = re.sub(r'([.!?])\s*([가-힣A-Z])', r'\1 \2', cleaned)
            
            return cleaned.strip()
            
        except Exception as e:
            logger.warning(f"텍스트 정리 실패: {e}")
            return content
    
    async def _chunk_document(
        self, 
        document: Document, 
        content: str, 
        structure_info: Dict[str, Any]
    ) -> List[Chunk]:
        """문서 청킹"""
        try:
            # 청킹 전략 결정
            strategy = self._determine_chunking_strategy(document, structure_info)
            
            log_with_context(
                logger, "debug",
                "청킹 전략 결정",
                document_id=document.id,
                strategy=strategy
            )
            
            # 청킹 수행
            chunks = await self.korean_chunker.chunk_document(
                document, content, structure_info, strategy
            )
            
            return chunks
            
        except Exception as e:
            logger.error(f"문서 청킹 실패: {e}")
            raise ChunkingError(f"청킹 실패: {e}")
    
    def _determine_chunking_strategy(
        self, 
        document: Document, 
        structure_info: Dict[str, Any]
    ) -> str:
        """청킹 전략 결정"""
        # 문서 타입별 기본 전략
        if document.document_type == DocumentType.REGULATION:
            if structure_info.get("has_articles", False):
                return "article_based"  # 조항 기반
            else:
                return "structural"
        
        elif document.document_type == DocumentType.CIVIL_AFFAIRS:
            if structure_info.get("has_items", False):
                return "item_based"  # 항목 기반
            else:
                return "semantic"
        
        elif document.document_type == DocumentType.ADMINISTRATIVE:
            if structure_info.get("has_sections", False):
                return "section_based"  # 절 기반
            else:
                return "semantic"
        
        # 기본 전략
        return self.config.strategy
    
    async def _post_process_chunks(
        self, 
        chunks: List[Chunk], 
        document: Document
    ) -> List[Chunk]:
        """청크 후처리"""
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # 청크 ID 생성
                chunk.id = f"{document.id}_chunk_{i:03d}"
                
                # 메타데이터 보강
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.update({
                    "document_id": document.id,
                    "document_type": document.document_type.value,
                    "document_title": document.title,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "processing_time": datetime.now().isoformat(),
                    "chunk_length": len(chunk.text),
                    "source_table": document.source_table
                })
                
                # 청크 품질 검증
                if self._validate_chunk_quality(chunk):
                    processed_chunks.append(chunk)
                else:
                    logger.warning(f"청크 품질 검증 실패: {chunk.id}")
                
            except Exception as e:
                logger.warning(f"청크 후처리 실패: {e}")
                continue
        
        return processed_chunks
    
    def _validate_chunk_quality(self, chunk: Chunk) -> bool:
        """청크 품질 검증"""
        # 최소 길이 검증
        if len(chunk.text.strip()) < self.config.min_chunk_size:
            return False
        
        # 최대 길이 검증
        if len(chunk.text) > self.config.max_chunk_size:
            return False
        
        # 의미있는 내용 포함 여부
        import re
        korean_chars = len(re.findall(r'[가-힣]', chunk.text))
        if korean_chars < 10:  # 최소 한국어 10글자
            return False
        
        return True
    
    def _update_processing_stats(
        self, 
        document: Document, 
        chunk_count: int, 
        processing_time: float
    ):
        """처리 통계 업데이트"""
        self.processing_stats["total_processed"] += 1
        self.processing_stats["successful_processing"] += 1
        self.processing_stats["total_chunks_created"] += chunk_count
        self.processing_stats["processing_time_total"] += processing_time
        self.processing_stats["type_distribution"][document.document_type.value] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """처리 통계 조회"""
        stats = self.processing_stats.copy()
        
        # 평균 처리 시간 계산
        if stats["successful_processing"] > 0:
            stats["avg_processing_time"] = (
                stats["processing_time_total"] / stats["successful_processing"]
            )
        else:
            stats["avg_processing_time"] = 0.0
        
        # 평균 청크 수 계산
        if stats["successful_processing"] > 0:
            stats["avg_chunks_per_document"] = (
                stats["total_chunks_created"] / stats["successful_processing"]
            )
        else:
            stats["avg_chunks_per_document"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "failed_processing": 0,
            "total_chunks_created": 0,
            "processing_time_total": 0.0,
            "type_distribution": {
                "행정문서": 0,
                "민원문서": 0,
                "조례문서": 0
            }
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            await self.structure_analyzer.cleanup()
            await self.korean_chunker.cleanup()
            await self.document_classifier.cleanup()
            
            self.is_initialized = False
            logger.info("문서 처리기 정리 완료")
            
        except Exception as e:
            logger.error(f"리소스 정리 실패: {e}")