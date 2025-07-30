"""
청킹 전략 기본 클래스
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from shared.models.document import Document, Chunk
from shared.models.config import ChunkingConfig
from shared.utils.logging import setup_logging

logger = setup_logging("base-chunker")


class BaseChunker(ABC):
    """청킹 전략 기본 클래스"""
    
    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.is_initialized = False
    
    async def initialize(self):
        """청킹 전략 초기화"""
        self.is_initialized = True
        logger.debug(f"{self.__class__.__name__} 초기화 완료")
    
    @abstractmethod
    async def chunk(
        self, 
        document: Document, 
        content: str, 
        structure_info: Dict[str, Any]
    ) -> List[Chunk]:
        """문서 청킹 (추상 메서드)"""
        pass
    
    def _create_chunk(
        self, 
        text: str, 
        document_id: str, 
        chunk_index: int,
        metadata: Dict[str, Any] = None
    ) -> Chunk:
        """청크 객체 생성"""
        return Chunk(
            id=f"{document_id}_chunk_{chunk_index:03d}",
            document_id=document_id,
            text=text.strip(),
            chunk_index=chunk_index,
            metadata=metadata or {}
        )
    
    def _validate_chunk_size(self, text: str) -> bool:
        """청크 크기 검증"""
        length = len(text.strip())
        return self.config.min_chunk_size <= length <= self.config.max_chunk_size
    
    def _split_oversized_chunk(self, text: str) -> List[str]:
        """크기 초과 청크 분할"""
        if len(text) <= self.config.max_chunk_size:
            return [text]
        
        # 문장 단위로 분할
        from shared.utils.korean_utils import split_korean_sentences
        sentences = split_korean_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def cleanup(self):
        """리소스 정리"""
        self.is_initialized = False
        logger.debug(f"{self.__class__.__name__} 정리 완료")