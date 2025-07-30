"""
한국어 텍스트 처리 유틸리티
"""
import re
from typing import List, Tuple


def clean_korean_text(text: str) -> str:
    """한국어 텍스트 정리"""
    if not text:
        return ""
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 특수문자 정리 (한국어 문서에서 자주 나타나는 패턴)
    text = re.sub(r'[^\w\s가-힣.,!?;:()\[\]{}""''「」『』\-]', '', text)
    
    # 앞뒤 공백 제거
    text = text.strip()
    
    return text


def split_korean_sentences(text: str) -> List[str]:
    """한국어 문장 분리"""
    if not text:
        return []
    
    # 한국어 문장 종결 패턴
    sentence_endings = r'[.!?](?=\s|$)|[다가나까요]\.(?=\s|$)'
    
    sentences = re.split(sentence_endings, text)
    
    # 빈 문장 제거 및 정리
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and len(sentence) > 5:  # 너무 짧은 문장 제외
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences


def extract_korean_keywords(text: str) -> List[str]:
    """한국어 키워드 추출 (간단한 규칙 기반)"""
    if not text:
        return []
    
    # 한국어 명사 패턴 (간단한 휴리스틱)
    noun_patterns = [
        r'[가-힣]{2,}(?=의|을|를|이|가|에|에서|으로|로)',  # 조사 앞의 명사
        r'[가-힣]{3,}(?=\s|$|[.,!?])',  # 3글자 이상 한국어 단어
    ]
    
    keywords = []
    for pattern in noun_patterns:
        matches = re.findall(pattern, text)
        keywords.extend(matches)
    
    # 중복 제거 및 정렬
    keywords = list(set(keywords))
    keywords.sort(key=len, reverse=True)
    
    return keywords[:10]  # 상위 10개만 반환


def detect_document_structure(text: str) -> dict:
    """문서 구조 감지 (조례, 행정문서 등)"""
    structure = {
        "has_articles": False,  # 조항이 있는지
        "has_items": False,     # 항목이 있는지
        "has_sections": False,  # 절이 있는지
        "document_type": "일반문서"
    }
    
    # 조항 패턴 감지
    article_patterns = [
        r'제\s*\d+\s*조',  # 제1조, 제 1 조
        r'제\s*[일이삼사오육칠팔구십]+\s*조',  # 제일조, 제이조
    ]
    
    for pattern in article_patterns:
        if re.search(pattern, text):
            structure["has_articles"] = True
            structure["document_type"] = "조례문서"
            break
    
    # 항목 패턴 감지
    item_patterns = [
        r'\d+\.\s',  # 1. 2. 3.
        r'[가나다라마바사아자차카타파하]\.\s',  # 가. 나. 다.
    ]
    
    for pattern in item_patterns:
        if re.search(pattern, text):
            structure["has_items"] = True
            break
    
    # 절 패턴 감지
    section_patterns = [
        r'제\s*\d+\s*절',  # 제1절
        r'제\s*[일이삼사오육칠팔구십]+\s*절',  # 제일절
    ]
    
    for pattern in section_patterns:
        if re.search(pattern, text):
            structure["has_sections"] = True
            break
    
    # 민원문서 패턴 감지
    civil_patterns = [
        r'민원|신청|접수|처리|승인|허가',
        r'신청서|접수증|처리결과|승인서'
    ]
    
    for pattern in civil_patterns:
        if re.search(pattern, text):
            if structure["document_type"] == "일반문서":
                structure["document_type"] = "민원문서"
            break
    
    # 행정문서 패턴 감지
    admin_patterns = [
        r'공문|시행|시달|통보|지시',
        r'업무|처리|보고|계획|예산'
    ]
    
    for pattern in admin_patterns:
        if re.search(pattern, text):
            if structure["document_type"] == "일반문서":
                structure["document_type"] = "행정문서"
            break
    
    return structure


def chunk_by_structure(text: str, structure: dict, max_chunk_size: int = 1000) -> List[Tuple[str, dict]]:
    """구조 기반 청킹"""
    chunks = []
    
    if structure["has_articles"]:
        # 조항 기반 분할
        article_pattern = r'(제\s*\d+\s*조[^제]*?)(?=제\s*\d+\s*조|$)'
        matches = re.finditer(article_pattern, text, re.DOTALL)
        
        for match in matches:
            chunk_text = match.group(1).strip()
            if len(chunk_text) > 50:  # 너무 짧은 청크 제외
                chunk_metadata = {
                    "chunk_type": "article",
                    "structure_info": "조항 단위"
                }
                
                # 청크가 너무 크면 추가 분할
                if len(chunk_text) > max_chunk_size:
                    sub_chunks = _split_large_chunk(chunk_text, max_chunk_size)
                    for i, sub_chunk in enumerate(sub_chunks):
                        sub_metadata = chunk_metadata.copy()
                        sub_metadata["sub_chunk_index"] = i
                        chunks.append((sub_chunk, sub_metadata))
                else:
                    chunks.append((chunk_text, chunk_metadata))
    
    elif structure["has_items"]:
        # 항목 기반 분할
        item_pattern = r'(\d+\.\s[^0-9]*?)(?=\d+\.\s|$)'
        matches = re.finditer(item_pattern, text, re.DOTALL)
        
        for match in matches:
            chunk_text = match.group(1).strip()
            if len(chunk_text) > 50:
                chunk_metadata = {
                    "chunk_type": "item",
                    "structure_info": "항목 단위"
                }
                chunks.append((chunk_text, chunk_metadata))
    
    else:
        # 문단 기반 분할
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(current_chunk + paragraph) > max_chunk_size:
                if current_chunk:
                    chunk_metadata = {
                        "chunk_type": "paragraph",
                        "structure_info": "문단 단위"
                    }
                    chunks.append((current_chunk.strip(), chunk_metadata))
                current_chunk = paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        # 마지막 청크 추가
        if current_chunk:
            chunk_metadata = {
                "chunk_type": "paragraph",
                "structure_info": "문단 단위"
            }
            chunks.append((current_chunk.strip(), chunk_metadata))
    
    return chunks


def _split_large_chunk(text: str, max_size: int) -> List[str]:
    """큰 청크를 문장 단위로 분할"""
    sentences = split_korean_sentences(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence) > max_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks