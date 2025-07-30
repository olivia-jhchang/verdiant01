"""
한국어 특화 청킹 시스템
"""
import re
import asyncio
from typing import List, Dict, Any, Tuple
from datetime import datetime

from shared.models.document import Document, Chunk, DocumentType
from shared.models.config import ChunkingConfig
from shared.models.exceptions import ChunkingError
from shared.utils.logging import setup_logging, log_with_context
from shared.utils.korean_utils import (
    split_korean_sentences,
    chunk_by_structure,
    detect_document_structure
)

from app.chunkers.strategies import (
    ArticleBasedChunker,
    ItemBasedChunker,
    SectionBasedChunker,
    SemanticChunker,
    SentenceBasedChunker
)

logger = setup_logging("korean-chunker")


class KoreanChunker:
    """한국어 특화 청킹 시스템"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.is_initialized = False
        
        # 청킹 전략별 처리기
        self.chunkers = {}
        
        # 청킹 통계
        self.chunking_stats = {
            "total_documents_chunked": 0,
            "total_chunks_created": 0,
            "strategy_usage": {},
            "avg_chunks_per_document": 0.0,
            "avg_chunk_size": 0.0,
            "failed_chunking": 0
        }
    
    async def initialize(self):
        """청킹 시스템 초기화"""
        try:
            # 각 전략별 청킹 처리기 초기화
            self.chunkers = {
                "article_based": ArticleBasedChunker(self.config),
                "item_based": ItemBasedChunker(self.config),
                "section_based": SectionBasedChunker(self.config),
                "semantic": SemanticChunker(self.config),
                "sentence": SentenceBasedChunker(self.config)
            }
            
            # 각 청킹 처리기 초기화
            for strategy, chunker in self.chunkers.items():
                await chunker.initialize()
                self.chunking_stats["strategy_usage"][strategy] = 0
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "한국어 청킹 시스템 초기화 완료",
                config=self.config.__dict__,
                available_strategies=list(self.chunkers.keys())
            )
            
        except Exception as e:
            logger.error(f"청킹 시스템 초기화 실패: {e}")
            raise ChunkingError(f"초기화 실패: {e}")
    
    async def chunk_document(
        self,
        document: Document,
        content: str,
        structure_info: Dict[str, Any],
        strategy: str = None
    ) -> List[Chunk]:
        """문서 청킹"""
        if not self.is_initialized:
            raise ChunkingError("청킹 시스템이 초기화되지 않았습니다")
        
        try:
            # 전략이 지정되지 않은 경우 자동 결정
            if not strategy:
                strategy = self._determine_optimal_strategy(document, structure_info)
            
            log_with_context(
                logger, "info",
                "문서 청킹 시작",
                document_id=document.id,
                strategy=strategy,
                content_length=len(content)
            )
            
            # 전략별 청킹 수행
            chunks = await self._execute_chunking_strategy(
                document, content, structure_info, strategy
            )
            
            # 청킹 결과 검증 및 후처리
            validated_chunks = await self._validate_and_post_process_chunks(
                chunks, document, strategy
            )
            
            # 통계 업데이트
            self._update_chunking_stats(strategy, len(validated_chunks), validated_chunks)
            
            log_with_context(
                logger, "info",
                "문서 청킹 완료",
                document_id=document.id,
                strategy=strategy,
                chunks_created=len(validated_chunks)
            )
            
            return validated_chunks
            
        except Exception as e:
            self.chunking_stats["failed_chunking"] += 1
            logger.error(f"문서 청킹 실패: {e}")
            
            # 대안 전략 시도
            try:
                return await self._fallback_chunking(document, content)
            except Exception as fallback_error:
                raise ChunkingError(f"청킹 실패 (대안 전략 포함): {fallback_error}")
    
    def _determine_optimal_strategy(
        self, 
        document: Document, 
        structure_info: Dict[str, Any]
    ) -> str:
        """최적 청킹 전략 결정"""
        # 문서 타입별 기본 전략
        if document.document_type == DocumentType.REGULATION:
            if structure_info.get("has_articles", False):
                return "article_based"
            elif structure_info.get("has_sections", False):
                return "section_based"
            else:
                return "semantic"
        
        elif document.document_type == DocumentType.CIVIL_AFFAIRS:
            if structure_info.get("has_items", False):
                return "item_based"
            else:
                return "semantic"
        
        elif document.document_type == DocumentType.ADMINISTRATIVE:
            if structure_info.get("has_sections", False):
                return "section_based"
            else:
                return "semantic"
        
        # 내용 길이 기반 전략 조정
        content_length = len(document.content)
        if content_length < 500:
            return "sentence"  # 짧은 문서는 문장 단위
        elif content_length > 5000:
            return "semantic"  # 긴 문서는 의미 단위
        
        return self.config.strategy or "semantic"
    
    async def _execute_chunking_strategy(
        self,
        document: Document,
        content: str,
        structure_info: Dict[str, Any],
        strategy: str
    ) -> List[Chunk]:
        """청킹 전략 실행"""
        if strategy not in self.chunkers:
            logger.warning(f"알 수 없는 청킹 전략: {strategy}, 기본 전략 사용")
            strategy = "semantic"
        
        chunker = self.chunkers[strategy]
        
        try:
            chunks = await chunker.chunk(document, content, structure_info)
            self.chunking_stats["strategy_usage"][strategy] += 1
            return chunks
            
        except Exception as e:
            logger.warning(f"청킹 전략 {strategy} 실행 실패: {e}")
            
            # 대안 전략 시도
            fallback_strategies = ["semantic", "sentence"]
            for fallback_strategy in fallback_strategies:
                if fallback_strategy != strategy and fallback_strategy in self.chunkers:
                    try:
                        logger.info(f"대안 청킹 전략 시도: {fallback_strategy}")
                        fallback_chunker = self.chunkers[fallback_strategy]
                        chunks = await fallback_chunker.chunk(document, content, structure_info)
                        self.chunking_stats["strategy_usage"][fallback_strategy] += 1
                        return chunks
                    except Exception as fallback_error:
                        logger.warning(f"대안 전략 {fallback_strategy} 실패: {fallback_error}")
                        continue
            
            raise ChunkingError(f"모든 청킹 전략 실패: {e}")
    
    async def _validate_and_post_process_chunks(
        self,
        chunks: List[Chunk],
        document: Document,
        strategy: str
    ) -> List[Chunk]:
        """청크 검증 및 후처리"""
        validated_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # 기본 검증
                if not self._validate_chunk_basic(chunk):
                    continue
                
                # 청크 ID 설정
                chunk.id = f"{document.id}_chunk_{i:03d}"
                chunk.document_id = document.id
                chunk.chunk_index = i
                
                # 메타데이터 보강
                chunk.metadata = chunk.metadata or {}
                chunk.metadata.update({
                    "chunking_strategy": strategy,
                    "chunk_created_at": datetime.now().isoformat(),
                    "chunk_length": len(chunk.text),
                    "document_type": document.document_type.value,
                    "source_table": document.source_table
                })
                
                # 청크 품질 점수 계산
                quality_score = self._calculate_chunk_quality_score(chunk)
                chunk.metadata["quality_score"] = quality_score
                
                # 품질 임계값 확인
                if quality_score >= 0.5:
                    validated_chunks.append(chunk)
                else:
                    logger.debug(f"청크 품질 점수 낮음: {chunk.id} (점수: {quality_score})")
                
            except Exception as e:
                logger.warning(f"청크 후처리 실패: {e}")
                continue
        
        # 청크가 너무 적은 경우 대안 처리
        if len(validated_chunks) < 2 and len(document.content) > 200:
            logger.warning(f"청크 수가 부족함 ({len(validated_chunks)}개), 대안 청킹 시도")
            return await self._fallback_chunking(document, document.content)
        
        return validated_chunks
    
    def _validate_chunk_basic(self, chunk: Chunk) -> bool:
        """기본 청크 검증"""
        if not chunk.text or not chunk.text.strip():
            return False
        
        text_length = len(chunk.text.strip())
        
        # 길이 검증
        if text_length < self.config.min_chunk_size:
            return False
        
        if text_length > self.config.max_chunk_size:
            return False
        
        # 의미있는 내용 포함 여부
        korean_chars = len(re.findall(r'[가-힣]', chunk.text))
        if korean_chars < 10:  # 최소 한국어 10글자
            return False
        
        # 너무 반복적인 내용 제외
        words = chunk.text.split()
        if len(set(words)) < len(words) * 0.3:  # 고유 단어 비율이 30% 미만
            return False
        
        return True
    
    def _calculate_chunk_quality_score(self, chunk: Chunk) -> float:
        """청크 품질 점수 계산"""
        score = 0.0
        text = chunk.text.strip()
        
        # 길이 점수 (0.3)
        length_score = self._calculate_length_score(len(text))
        score += length_score * 0.3
        
        # 내용 다양성 점수 (0.2)
        diversity_score = self._calculate_diversity_score(text)
        score += diversity_score * 0.2
        
        # 구조적 완결성 점수 (0.2)
        completeness_score = self._calculate_completeness_score(text)
        score += completeness_score * 0.2
        
        # 한국어 비율 점수 (0.2)
        korean_score = self._calculate_korean_ratio_score(text)
        score += korean_score * 0.2
        
        # 문장 완결성 점수 (0.1)
        sentence_score = self._calculate_sentence_completeness_score(text)
        score += sentence_score * 0.1
        
        return min(score, 1.0)
    
    def _calculate_length_score(self, length: int) -> float:
        """길이 기반 점수"""
        optimal_length = (self.config.min_chunk_size + self.config.max_chunk_size) / 2
        
        if length == optimal_length:
            return 1.0
        
        # 최적 길이에서 멀어질수록 점수 감소
        distance = abs(length - optimal_length)
        max_distance = max(
            optimal_length - self.config.min_chunk_size,
            self.config.max_chunk_size - optimal_length
        )
        
        return max(0.0, 1.0 - (distance / max_distance))
    
    def _calculate_diversity_score(self, text: str) -> float:
        """내용 다양성 점수"""
        words = text.split()
        if not words:
            return 0.0
        
        unique_words = set(words)
        diversity_ratio = len(unique_words) / len(words)
        
        return min(diversity_ratio * 2, 1.0)  # 50% 이상이면 만점
    
    def _calculate_completeness_score(self, text: str) -> float:
        """구조적 완결성 점수"""
        # 문장 종결 확인
        if text.endswith(('.', '!', '?', '다', '요', '음')):
            sentence_end_score = 1.0
        else:
            sentence_end_score = 0.5
        
        # 문단 구조 확인
        has_paragraph_structure = '\n' in text or len(text) > 100
        paragraph_score = 1.0 if has_paragraph_structure else 0.7
        
        return (sentence_end_score + paragraph_score) / 2
    
    def _calculate_korean_ratio_score(self, text: str) -> float:
        """한국어 비율 점수"""
        korean_chars = len(re.findall(r'[가-힣]', text))
        total_chars = len(re.findall(r'[가-힣a-zA-Z0-9]', text))
        
        if total_chars == 0:
            return 0.0
        
        korean_ratio = korean_chars / total_chars
        return korean_ratio  # 한국어 비율이 높을수록 좋음
    
    def _calculate_sentence_completeness_score(self, text: str) -> float:
        """문장 완결성 점수"""
        sentences = split_korean_sentences(text)
        if not sentences:
            return 0.0
        
        complete_sentences = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and (
                sentence.endswith(('.', '!', '?')) or
                sentence.endswith(('다', '요', '음', '니다', '습니다'))
            ):
                complete_sentences += 1
        
        return complete_sentences / len(sentences)
    
    async def _fallback_chunking(self, document: Document, content: str) -> List[Chunk]:
        """대안 청킹 (문장 기반)"""
        logger.info(f"대안 청킹 수행: {document.id}")
        
        try:
            # 문장 단위로 분할
            sentences = split_korean_sentences(content)
            
            chunks = []
            current_chunk_text = ""
            chunk_index = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # 현재 청크에 문장 추가 시 크기 확인
                potential_text = current_chunk_text + " " + sentence if current_chunk_text else sentence
                
                if len(potential_text) <= self.config.max_chunk_size:
                    current_chunk_text = potential_text
                else:
                    # 현재 청크 완성
                    if current_chunk_text and len(current_chunk_text) >= self.config.min_chunk_size:
                        chunk = Chunk(
                            id=f"{document.id}_fallback_chunk_{chunk_index:03d}",
                            document_id=document.id,
                            text=current_chunk_text,
                            chunk_index=chunk_index,
                            metadata={
                                "chunking_strategy": "fallback_sentence",
                                "is_fallback": True
                            }
                        )
                        chunks.append(chunk)
                        chunk_index += 1
                    
                    # 새 청크 시작
                    current_chunk_text = sentence
            
            # 마지막 청크 처리
            if current_chunk_text and len(current_chunk_text) >= self.config.min_chunk_size:
                chunk = Chunk(
                    id=f"{document.id}_fallback_chunk_{chunk_index:03d}",
                    document_id=document.id,
                    text=current_chunk_text,
                    chunk_index=chunk_index,
                    metadata={
                        "chunking_strategy": "fallback_sentence",
                        "is_fallback": True
                    }
                )
                chunks.append(chunk)
            
            logger.info(f"대안 청킹 완료: {len(chunks)}개 청크 생성")
            return chunks
            
        except Exception as e:
            logger.error(f"대안 청킹 실패: {e}")
            raise ChunkingError(f"대안 청킹 실패: {e}")
    
    def _update_chunking_stats(
        self, 
        strategy: str, 
        chunk_count: int, 
        chunks: List[Chunk]
    ):
        """청킹 통계 업데이트"""
        self.chunking_stats["total_documents_chunked"] += 1
        self.chunking_stats["total_chunks_created"] += chunk_count
        
        # 평균 청크 수 계산
        self.chunking_stats["avg_chunks_per_document"] = (
            self.chunking_stats["total_chunks_created"] / 
            self.chunking_stats["total_documents_chunked"]
        )
        
        # 평균 청크 크기 계산
        if chunks:
            total_chunk_size = sum(len(chunk.text) for chunk in chunks)
            current_avg = total_chunk_size / len(chunks)
            
            # 누적 평균 계산
            total_docs = self.chunking_stats["total_documents_chunked"]
            prev_avg = self.chunking_stats["avg_chunk_size"]
            
            self.chunking_stats["avg_chunk_size"] = (
                (prev_avg * (total_docs - 1) + current_avg) / total_docs
            )
    
    def get_chunking_stats(self) -> Dict[str, Any]:
        """청킹 통계 조회"""
        return self.chunking_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.chunking_stats = {
            "total_documents_chunked": 0,
            "total_chunks_created": 0,
            "strategy_usage": {strategy: 0 for strategy in self.chunkers.keys()},
            "avg_chunks_per_document": 0.0,
            "avg_chunk_size": 0.0,
            "failed_chunking": 0
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            for chunker in self.chunkers.values():
                await chunker.cleanup()
            
            self.chunkers = {}
            self.is_initialized = False
            logger.info("한국어 청킹 시스템 정리 완료")
            
        except Exception as e:
            logger.error(f"청킹 시스템 정리 실패: {e}")