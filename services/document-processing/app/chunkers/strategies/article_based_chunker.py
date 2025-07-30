"""
조항 기반 청킹 전략
"""
import re
from typing import List, Dict, Any
from shared.models.document import Document, Chunk
from shared.utils.logging import setup_logging
from .base_chunker import BaseChunker

logger = setup_logging("article-chunker")


class ArticleBasedChunker(BaseChunker):
    """조항 기반 청킹 전략 (조례문서용)"""
    
    async def initialize(self):
        """조항 기반 청킹 초기화"""
        await super().initialize()
        
        # 조항 패턴 정의
        self.article_patterns = [
            re.compile(r'(제\s*\d+\s*조\s*\([^)]+\)[^제]*?)(?=제\s*\d+\s*조|$)', re.DOTALL),
            re.compile(r'(제\s*\d+\s*조[^제]*?)(?=제\s*\d+\s*조|$)', re.DOTALL),
            re.compile(r'(제\s*[일이삼사오육칠팔구십]+\s*조[^제]*?)(?=제\s*[일이삼사오육칠팔구십]+\s*조|$)', re.DOTALL)
        ]
        
        logger.debug("조항 기반 청킹 초기화 완료")
    
    async def chunk(
        self, 
        document: Document, 
        content: str, 
        structure_info: Dict[str, Any]
    ) -> List[Chunk]:
        """조항 기반 청킹 수행"""
        chunks = []
        chunk_index = 0
        
        # 조항별로 분할
        articles = self._extract_articles(content)
        
        if not articles:
            # 조항이 없으면 대안 전략 사용
            logger.warning(f"조항을 찾을 수 없음: {document.id}")
            return await self._fallback_to_paragraph_chunking(document, content)
        
        for article_text, article_info in articles:
            try:
                # 조항 텍스트 정리
                cleaned_text = self._clean_article_text(article_text)
                
                if not cleaned_text or len(cleaned_text) < self.config.min_chunk_size:
                    continue
                
                # 크기가 너무 큰 조항은 분할
                if len(cleaned_text) > self.config.max_chunk_size:
                    sub_chunks = self._split_large_article(cleaned_text, article_info)
                    for sub_chunk_text in sub_chunks:
                        if self._validate_chunk_size(sub_chunk_text):
                            chunk = self._create_chunk(
                                sub_chunk_text,
                                document.id,
                                chunk_index,
                                {
                                    "chunk_type": "article_sub",
                                    "article_info": article_info,
                                    "original_article_length": len(cleaned_text)
                                }
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                else:
                    # 적절한 크기의 조항은 그대로 청크로 사용
                    chunk = self._create_chunk(
                        cleaned_text,
                        document.id,
                        chunk_index,
                        {
                            "chunk_type": "article",
                            "article_info": article_info
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                    
            except Exception as e:
                logger.warning(f"조항 처리 실패: {e}")
                continue
        
        logger.info(f"조항 기반 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _extract_articles(self, content: str) -> List[tuple]:
        """조항 추출"""
        articles = []
        
        for pattern in self.article_patterns:
            matches = pattern.finditer(content)
            for match in matches:
                article_text = match.group(1).strip()
                
                # 조항 정보 추출
                article_info = self._parse_article_info(article_text)
                
                if article_info and len(article_text) > 50:  # 최소 길이 확인
                    articles.append((article_text, article_info))
        
        # 위치순으로 정렬
        articles.sort(key=lambda x: content.find(x[0]))
        
        # 중복 제거
        unique_articles = []
        seen_texts = set()
        
        for article_text, article_info in articles:
            if article_text not in seen_texts:
                unique_articles.append((article_text, article_info))
                seen_texts.add(article_text)
        
        return unique_articles
    
    def _parse_article_info(self, article_text: str) -> Dict[str, Any]:
        """조항 정보 파싱"""
        info = {}
        
        # 조항 번호 추출
        article_num_pattern = re.compile(r'제\s*(\d+)\s*조')
        article_num_match = article_num_pattern.search(article_text)
        if article_num_match:
            info["article_number"] = int(article_num_match.group(1))
        
        # 조항 제목 추출 (괄호 안의 내용)
        title_pattern = re.compile(r'제\s*\d+\s*조\s*\(([^)]+)\)')
        title_match = title_pattern.search(article_text)
        if title_match:
            info["article_title"] = title_match.group(1).strip()
        
        # 항목 수 계산
        item_pattern = re.compile(r'^\s*\d+\.\s', re.MULTILINE)
        item_matches = item_pattern.findall(article_text)
        info["item_count"] = len(item_matches)
        
        return info
    
    def _clean_article_text(self, article_text: str) -> str:
        """조항 텍스트 정리"""
        # 불필요한 공백 제거
        cleaned = re.sub(r'\s+', ' ', article_text)
        
        # 앞뒤 공백 제거
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _split_large_article(self, article_text: str, article_info: Dict[str, Any]) -> List[str]:
        """큰 조항 분할"""
        # 항목이 있는 경우 항목별로 분할
        if article_info.get("item_count", 0) > 0:
            return self._split_by_items(article_text)
        
        # 항목이 없는 경우 문단별로 분할
        return self._split_by_paragraphs(article_text)
    
    def _split_by_items(self, article_text: str) -> List[str]:
        """항목별 분할"""
        # 조항 제목 부분과 항목 부분 분리
        item_pattern = re.compile(r'(\d+\.\s[^0-9]*?)(?=\d+\.\s|$)', re.DOTALL)
        items = item_pattern.findall(article_text)
        
        # 조항 제목 부분 추출
        first_item_match = re.search(r'\d+\.\s', article_text)
        if first_item_match:
            header = article_text[:first_item_match.start()].strip()
        else:
            header = ""
        
        chunks = []
        
        # 헤더가 있고 충분히 긴 경우 별도 청크로 생성
        if header and len(header) >= self.config.min_chunk_size:
            chunks.append(header)
        
        # 항목들을 적절한 크기로 그룹화
        current_chunk = header if header and len(header) < self.config.min_chunk_size else ""
        
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            potential_chunk = current_chunk + "\n" + item if current_chunk else item
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = item
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, article_text: str) -> List[str]:
        """문단별 분할"""
        paragraphs = article_text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    async def _fallback_to_paragraph_chunking(
        self, 
        document: Document, 
        content: str
    ) -> List[Chunk]:
        """문단 기반 대안 청킹"""
        logger.info(f"조항 기반 청킹 실패, 문단 기반으로 대체: {document.id}")
        
        chunks = []
        paragraphs = content.split('\n\n')
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            potential_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            
            if len(potential_chunk) <= self.config.max_chunk_size:
                current_chunk = potential_chunk
            else:
                if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
                    chunk = self._create_chunk(
                        current_chunk,
                        document.id,
                        chunk_index,
                        {"chunk_type": "paragraph_fallback"}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = paragraph
        
        # 마지막 청크 처리
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk,
                document.id,
                chunk_index,
                {"chunk_type": "paragraph_fallback"}
            )
            chunks.append(chunk)
        
        return chunks