"""
의미 기반 청킹 전략
"""
import re
from typing import List, Dict, Any
from shared.models.document import Document, Chunk
from shared.utils.logging import setup_logging
from shared.utils.korean_utils import split_korean_sentences
from .base_chunker import BaseChunker

logger = setup_logging("semantic-chunker")


class SemanticChunker(BaseChunker):
    """의미 기반 청킹 전략"""
    
    async def initialize(self):
        """의미 기반 청킹 초기화"""
        await super().initialize()
        
        # 의미 단위 구분 패턴
        self.semantic_boundaries = [
            # 주제 변경 패턴
            re.compile(r'\n\s*[1-9]\d*\.\s+', re.MULTILINE),  # 1. 2. 3.
            re.compile(r'\n\s*[가나다라마바사아자차카타파하]\.\s+', re.MULTILINE),  # 가. 나. 다.
            re.compile(r'\n\s*[IVX]+\.\s+', re.MULTILINE),  # I. II. III.
            
            # 문단 구분
            re.compile(r'\n\s*\n\s*', re.MULTILINE),  # 빈 줄
            
            # 의미적 전환 표현
            re.compile(r'[.!?]\s*(?:그런데|하지만|그러나|따라서|그러므로|또한|한편|반면)', re.IGNORECASE),
            re.compile(r'[.!?]\s*(?:첫째|둘째|셋째|마지막으로|결론적으로)', re.IGNORECASE),
        ]
        
        logger.debug("의미 기반 청킹 초기화 완료")
    
    async def chunk(
        self, 
        document: Document, 
        content: str, 
        structure_info: Dict[str, Any]
    ) -> List[Chunk]:
        """의미 기반 청킹 수행"""
        chunks = []
        
        # 의미 단위로 분할
        semantic_units = self._extract_semantic_units(content)
        
        if not semantic_units:
            # 의미 단위를 찾을 수 없으면 문장 기반으로 대체
            return await self._fallback_to_sentence_chunking(document, content)
        
        # 의미 단위들을 적절한 크기의 청크로 조합
        chunks = self._combine_semantic_units(semantic_units, document.id)
        
        logger.info(f"의미 기반 청킹 완료: {len(chunks)}개 청크 생성")
        return chunks
    
    def _extract_semantic_units(self, content: str) -> List[Dict[str, Any]]:
        """의미 단위 추출"""
        units = []
        
        # 모든 경계점 찾기
        boundaries = []
        
        for pattern in self.semantic_boundaries:
            for match in pattern.finditer(content):
                boundaries.append({
                    'position': match.start(),
                    'type': self._classify_boundary_type(match.group()),
                    'strength': self._calculate_boundary_strength(match.group())
                })
        
        # 위치순으로 정렬
        boundaries.sort(key=lambda x: x['position'])
        
        # 중복 제거 (가까운 위치의 경계점들)
        filtered_boundaries = self._filter_close_boundaries(boundaries)
        
        # 의미 단위 생성
        start_pos = 0
        for boundary in filtered_boundaries:
            end_pos = boundary['position']
            
            if end_pos > start_pos:
                unit_text = content[start_pos:end_pos].strip()
                if unit_text and len(unit_text) > 20:  # 최소 길이
                    units.append({
                        'text': unit_text,
                        'start_pos': start_pos,
                        'end_pos': end_pos,
                        'boundary_type': boundary.get('type', 'unknown'),
                        'boundary_strength': boundary.get('strength', 0.5)
                    })
            
            start_pos = end_pos
        
        # 마지막 단위 처리
        if start_pos < len(content):
            unit_text = content[start_pos:].strip()
            if unit_text and len(unit_text) > 20:
                units.append({
                    'text': unit_text,
                    'start_pos': start_pos,
                    'end_pos': len(content),
                    'boundary_type': 'end',
                    'boundary_strength': 1.0
                })
        
        return units
    
    def _classify_boundary_type(self, boundary_text: str) -> str:
        """경계 타입 분류"""
        if re.search(r'[1-9]\d*\.', boundary_text):
            return 'numbered_list'
        elif re.search(r'[가나다라마바사아자차카타파하]\.', boundary_text):
            return 'korean_list'
        elif re.search(r'[IVX]+\.', boundary_text):
            return 'roman_list'
        elif re.search(r'\n\s*\n', boundary_text):
            return 'paragraph'
        elif re.search(r'그런데|하지만|그러나', boundary_text):
            return 'contrast'
        elif re.search(r'따라서|그러므로', boundary_text):
            return 'conclusion'
        elif re.search(r'또한|한편', boundary_text):
            return 'addition'
        else:
            return 'unknown'
    
    def _calculate_boundary_strength(self, boundary_text: str) -> float:
        """경계 강도 계산"""
        boundary_type = self._classify_boundary_type(boundary_text)
        
        strength_map = {
            'numbered_list': 0.9,
            'korean_list': 0.8,
            'roman_list': 0.9,
            'paragraph': 0.6,
            'contrast': 0.7,
            'conclusion': 0.8,
            'addition': 0.5,
            'unknown': 0.3
        }
        
        return strength_map.get(boundary_type, 0.5)
    
    def _filter_close_boundaries(self, boundaries: List[Dict]) -> List[Dict]:
        """가까운 경계점들 필터링"""
        if not boundaries:
            return []
        
        filtered = [boundaries[0]]
        min_distance = 50  # 최소 50자 간격
        
        for boundary in boundaries[1:]:
            last_boundary = filtered[-1]
            
            if boundary['position'] - last_boundary['position'] >= min_distance:
                filtered.append(boundary)
            else:
                # 더 강한 경계점으로 교체
                if boundary['strength'] > last_boundary['strength']:
                    filtered[-1] = boundary
        
        return filtered
    
    def _combine_semantic_units(self, units: List[Dict], document_id: str) -> List[Chunk]:
        """의미 단위들을 청크로 조합"""
        chunks = []
        current_chunk_text = ""
        current_units = []
        chunk_index = 0
        
        for unit in units:
            unit_text = unit['text']
            
            # 현재 청크에 단위 추가 시 크기 확인
            potential_text = current_chunk_text + "\n\n" + unit_text if current_chunk_text else unit_text
            
            if len(potential_text) <= self.config.max_chunk_size:
                current_chunk_text = potential_text
                current_units.append(unit)
            else:
                # 현재 청크 완성
                if current_chunk_text and len(current_chunk_text) >= self.config.min_chunk_size:
                    chunk = self._create_semantic_chunk(
                        current_chunk_text,
                        document_id,
                        chunk_index,
                        current_units
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # 새 청크 시작
                current_chunk_text = unit_text
                current_units = [unit]
        
        # 마지막 청크 처리
        if current_chunk_text and len(current_chunk_text) >= self.config.min_chunk_size:
            chunk = self._create_semantic_chunk(
                current_chunk_text,
                document_id,
                chunk_index,
                current_units
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_semantic_chunk(
        self, 
        text: str, 
        document_id: str, 
        chunk_index: int,
        units: List[Dict]
    ) -> Chunk:
        """의미 기반 청크 생성"""
        # 단위들의 경계 타입 분석
        boundary_types = [unit.get('boundary_type', 'unknown') for unit in units]
        avg_strength = sum(unit.get('boundary_strength', 0.5) for unit in units) / len(units)
        
        metadata = {
            'chunk_type': 'semantic',
            'semantic_units_count': len(units),
            'boundary_types': boundary_types,
            'avg_boundary_strength': avg_strength,
            'coherence_score': self._calculate_coherence_score(text)
        }
        
        return self._create_chunk(text, document_id, chunk_index, metadata)
    
    def _calculate_coherence_score(self, text: str) -> float:
        """텍스트 일관성 점수 계산"""
        sentences = split_korean_sentences(text)
        if len(sentences) < 2:
            return 1.0
        
        # 간단한 일관성 측정 (키워드 중복도 기반)
        all_words = []
        sentence_words = []
        
        for sentence in sentences:
            words = re.findall(r'[가-힣]{2,}', sentence)  # 한국어 단어 추출
            sentence_words.append(set(words))
            all_words.extend(words)
        
        if not all_words:
            return 0.5
        
        # 문장 간 공통 단어 비율 계산
        common_word_scores = []
        for i in range(len(sentence_words) - 1):
            common_words = sentence_words[i] & sentence_words[i + 1]
            total_words = sentence_words[i] | sentence_words[i + 1]
            
            if total_words:
                score = len(common_words) / len(total_words)
                common_word_scores.append(score)
        
        if common_word_scores:
            return sum(common_word_scores) / len(common_word_scores)
        else:
            return 0.5
    
    async def _fallback_to_sentence_chunking(
        self, 
        document: Document, 
        content: str
    ) -> List[Chunk]:
        """문장 기반 대안 청킹"""
        logger.info(f"의미 기반 청킹 실패, 문장 기반으로 대체: {document.id}")
        
        chunks = []
        sentences = split_korean_sentences(content)
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
                        current_chunk,
                        document.id,
                        chunk_index,
                        {"chunk_type": "sentence_fallback"}
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = sentence
        
        # 마지막 청크 처리
        if current_chunk and len(current_chunk) >= self.config.min_chunk_size:
            chunk = self._create_chunk(
                current_chunk,
                document.id,
                chunk_index,
                {"chunk_type": "sentence_fallback"}
            )
            chunks.append(chunk)
        
        return chunks