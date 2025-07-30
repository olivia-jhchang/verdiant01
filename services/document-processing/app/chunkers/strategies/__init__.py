# Chunking strategies
from .base_chunker import BaseChunker
from .article_based_chunker import ArticleBasedChunker
from .semantic_chunker import SemanticChunker

# 간단한 구현체들
class ItemBasedChunker(BaseChunker):
    """항목 기반 청킹 (민원문서용)"""
    async def chunk(self, document, content, structure_info):
        # 간단한 항목 기반 구현
        import re
        chunks = []
        items = re.split(r'\n\s*\d+\)\s*', content)
        
        for i, item in enumerate(items):
            if item.strip() and len(item.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    item.strip(), document.id, i, {"chunk_type": "item"}
                )
                chunks.append(chunk)
        
        return chunks if chunks else await self._fallback_chunking(document, content)
    
    async def _fallback_chunking(self, document, content):
        from shared.utils.korean_utils import split_korean_sentences
        sentences = split_korean_sentences(content)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk + sentence) <= self.config.max_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, document.id, chunk_index, {"chunk_type": "sentence"}))
                    chunk_index += 1
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, document.id, chunk_index, {"chunk_type": "sentence"}))
        
        return chunks


class SectionBasedChunker(BaseChunker):
    """절 기반 청킹 (행정문서용)"""
    async def chunk(self, document, content, structure_info):
        # 간단한 절 기반 구현
        import re
        chunks = []
        sections = re.split(r'\n\s*[IVX]+\.\s*', content)
        
        for i, section in enumerate(sections):
            if section.strip() and len(section.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    section.strip(), document.id, i, {"chunk_type": "section"}
                )
                chunks.append(chunk)
        
        return chunks if chunks else await self._fallback_chunking(document, content)
    
    async def _fallback_chunking(self, document, content):
        paragraphs = content.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= self.config.max_chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
            else:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunks.append(self._create_chunk(current_chunk, document.id, chunk_index, {"chunk_type": "paragraph"}))
                    chunk_index += 1
                current_chunk = paragraph
        
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunks.append(self._create_chunk(current_chunk, document.id, chunk_index, {"chunk_type": "paragraph"}))
        
        return chunks


class SentenceBasedChunker(BaseChunker):
    """문장 기반 청킹"""
    async def chunk(self, document, content, structure_info):
        from shared.utils.korean_utils import split_korean_sentences
        sentences = split_korean_sentences(content)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk, document.id, chunk_index, {"chunk_type": "sentence"}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                current_chunk = sentence
        
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk, document.id, chunk_index, {"chunk_type": "sentence"}
            )
            chunks.append(chunk)
        
        return chunks