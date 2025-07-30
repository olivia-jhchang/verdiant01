"""
문서 구조 분석기
"""
import re
from typing import Dict, Any, List, Tuple
from shared.models.document import DocumentType
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("structure-analyzer")


class StructureAnalyzer:
    """문서 구조 분석 클래스"""
    
    def __init__(self):
        self.is_initialized = False
        
        # 구조 패턴 정의
        self.patterns = {
            "regulation": {
                "articles": [
                    r'제\s*(\d+)\s*조\s*\(([^)]+)\)',  # 제1조 (목적)
                    r'제\s*(\d+)\s*조\s*([^\n]+)',     # 제1조 목적
                    r'제\s*([일이삼사오육칠팔구십]+)\s*조',  # 제일조
                ],
                "sections": [
                    r'제\s*(\d+)\s*절\s*([^\n]+)',     # 제1절
                    r'제\s*([일이삼사오육칠팔구십]+)\s*절',  # 제일절
                ],
                "chapters": [
                    r'제\s*(\d+)\s*장\s*([^\n]+)',     # 제1장
                    r'제\s*([일이삼사오육칠팔구십]+)\s*장',  # 제일장
                ],
                "items": [
                    r'(\d+)\.\s*([^\n]+)',             # 1. 항목
                    r'([가나다라마바사아자차카타파하])\.\s*([^\n]+)',  # 가. 항목
                ]
            },
            "administrative": {
                "sections": [
                    r'([IVX]+)\.\s*([^\n]+)',          # I. 개요
                    r'(\d+)\.\s*([^\n]+)',             # 1. 개요
                ],
                "subsections": [
                    r'(\d+)-(\d+)\.\s*([^\n]+)',       # 1-1. 세부사항
                    r'([가나다라마바사아자차카타파하])\)\s*([^\n]+)',  # 가) 세부사항
                ],
                "items": [
                    r'[-•]\s*([^\n]+)',                # - 항목, • 항목
                    r'○\s*([^\n]+)',                   # ○ 항목
                ]
            },
            "civil_affairs": {
                "procedures": [
                    r'(\d+)\s*단계\s*[:：]\s*([^\n]+)',  # 1단계: 신청
                    r'단계\s*(\d+)\s*[:：]\s*([^\n]+)',  # 단계1: 신청
                ],
                "requirements": [
                    r'필요서류\s*[:：]\s*([^\n]+)',      # 필요서류:
                    r'구비서류\s*[:：]\s*([^\n]+)',      # 구비서류:
                    r'제출서류\s*[:：]\s*([^\n]+)',      # 제출서류:
                ],
                "items": [
                    r'(\d+)\)\s*([^\n]+)',             # 1) 항목
                    r'([가나다라마바사아자차카타파하])\)\s*([^\n]+)',  # 가) 항목
                ]
            }
        }
    
    async def initialize(self):
        """구조 분석기 초기화"""
        try:
            # 패턴 컴파일
            self.compiled_patterns = {}
            for doc_type, type_patterns in self.patterns.items():
                self.compiled_patterns[doc_type] = {}
                for pattern_type, patterns in type_patterns.items():
                    self.compiled_patterns[doc_type][pattern_type] = [
                        re.compile(pattern, re.MULTILINE | re.IGNORECASE)
                        for pattern in patterns
                    ]
            
            self.is_initialized = True
            logger.info("구조 분석기 초기화 완료")
            
        except Exception as e:
            logger.error(f"구조 분석기 초기화 실패: {e}")
            raise e
    
    async def analyze_structure(
        self, 
        content: str, 
        document_type: DocumentType
    ) -> Dict[str, Any]:
        """문서 구조 상세 분석"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 문서 타입에 따른 분석
            if document_type == DocumentType.REGULATION:
                return await self._analyze_regulation_structure(content)
            elif document_type == DocumentType.ADMINISTRATIVE:
                return await self._analyze_administrative_structure(content)
            elif document_type == DocumentType.CIVIL_AFFAIRS:
                return await self._analyze_civil_affairs_structure(content)
            else:
                return await self._analyze_general_structure(content)
                
        except Exception as e:
            logger.warning(f"구조 분석 실패: {e}")
            return {"structure_type": "unknown", "elements": []}
    
    async def _analyze_regulation_structure(self, content: str) -> Dict[str, Any]:
        """조례문서 구조 분석"""
        structure = {
            "structure_type": "regulation",
            "elements": [],
            "hierarchy": [],
            "statistics": {}
        }
        
        # 장(章) 분석
        chapters = self._extract_elements(
            content, self.compiled_patterns["regulation"]["chapters"]
        )
        
        # 절(節) 분석
        sections = self._extract_elements(
            content, self.compiled_patterns["regulation"]["sections"]
        )
        
        # 조(條) 분석
        articles = self._extract_elements(
            content, self.compiled_patterns["regulation"]["articles"]
        )
        
        # 항목 분석
        items = self._extract_elements(
            content, self.compiled_patterns["regulation"]["items"]
        )
        
        # 구조 요소 정리
        structure["elements"] = {
            "chapters": chapters,
            "sections": sections,
            "articles": articles,
            "items": items
        }
        
        # 계층 구조 구성
        structure["hierarchy"] = self._build_regulation_hierarchy(
            chapters, sections, articles, items
        )
        
        # 통계 정보
        structure["statistics"] = {
            "chapter_count": len(chapters),
            "section_count": len(sections),
            "article_count": len(articles),
            "item_count": len(items),
            "has_hierarchical_structure": len(articles) > 0
        }
        
        log_with_context(
            logger, "debug",
            "조례문서 구조 분석 완료",
            statistics=structure["statistics"]
        )
        
        return structure
    
    async def _analyze_administrative_structure(self, content: str) -> Dict[str, Any]:
        """행정문서 구조 분석"""
        structure = {
            "structure_type": "administrative",
            "elements": [],
            "hierarchy": [],
            "statistics": {}
        }
        
        # 대분류 섹션 분석
        sections = self._extract_elements(
            content, self.compiled_patterns["administrative"]["sections"]
        )
        
        # 소분류 섹션 분석
        subsections = self._extract_elements(
            content, self.compiled_patterns["administrative"]["subsections"]
        )
        
        # 항목 분석
        items = self._extract_elements(
            content, self.compiled_patterns["administrative"]["items"]
        )
        
        # 구조 요소 정리
        structure["elements"] = {
            "sections": sections,
            "subsections": subsections,
            "items": items
        }
        
        # 계층 구조 구성
        structure["hierarchy"] = self._build_administrative_hierarchy(
            sections, subsections, items
        )
        
        # 통계 정보
        structure["statistics"] = {
            "section_count": len(sections),
            "subsection_count": len(subsections),
            "item_count": len(items),
            "has_structured_format": len(sections) > 0
        }
        
        return structure
    
    async def _analyze_civil_affairs_structure(self, content: str) -> Dict[str, Any]:
        """민원문서 구조 분석"""
        structure = {
            "structure_type": "civil_affairs",
            "elements": [],
            "hierarchy": [],
            "statistics": {}
        }
        
        # 절차 분석
        procedures = self._extract_elements(
            content, self.compiled_patterns["civil_affairs"]["procedures"]
        )
        
        # 필요서류 분석
        requirements = self._extract_elements(
            content, self.compiled_patterns["civil_affairs"]["requirements"]
        )
        
        # 항목 분석
        items = self._extract_elements(
            content, self.compiled_patterns["civil_affairs"]["items"]
        )
        
        # 구조 요소 정리
        structure["elements"] = {
            "procedures": procedures,
            "requirements": requirements,
            "items": items
        }
        
        # 특별 패턴 분석
        special_patterns = self._analyze_civil_affairs_patterns(content)
        structure["special_patterns"] = special_patterns
        
        # 통계 정보
        structure["statistics"] = {
            "procedure_count": len(procedures),
            "requirement_count": len(requirements),
            "item_count": len(items),
            "has_procedure_structure": len(procedures) > 0,
            **special_patterns
        }
        
        return structure
    
    async def _analyze_general_structure(self, content: str) -> Dict[str, Any]:
        """일반 문서 구조 분석"""
        structure = {
            "structure_type": "general",
            "elements": [],
            "statistics": {}
        }
        
        # 기본 구조 요소 분석
        paragraphs = self._extract_paragraphs(content)
        sentences = self._extract_sentences(content)
        
        structure["elements"] = {
            "paragraphs": paragraphs,
            "sentences": sentences
        }
        
        structure["statistics"] = {
            "paragraph_count": len(paragraphs),
            "sentence_count": len(sentences),
            "avg_paragraph_length": sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
            "avg_sentence_length": sum(len(s) for s in sentences) / len(sentences) if sentences else 0
        }
        
        return structure
    
    def _extract_elements(
        self, 
        content: str, 
        patterns: List[re.Pattern]
    ) -> List[Dict[str, Any]]:
        """패턴을 사용하여 구조 요소 추출"""
        elements = []
        
        for pattern in patterns:
            matches = pattern.finditer(content)
            for match in matches:
                element = {
                    "text": match.group(0),
                    "position": match.span(),
                    "groups": match.groups()
                }
                elements.append(element)
        
        # 위치순으로 정렬
        elements.sort(key=lambda x: x["position"][0])
        
        return elements
    
    def _extract_paragraphs(self, content: str) -> List[str]:
        """문단 추출"""
        paragraphs = content.split('\n\n')
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _extract_sentences(self, content: str) -> List[str]:
        """문장 추출"""
        # 한국어 문장 종결 패턴
        sentence_pattern = re.compile(r'[^.!?]*[.!?]')
        sentences = sentence_pattern.findall(content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _build_regulation_hierarchy(
        self, 
        chapters: List[Dict], 
        sections: List[Dict], 
        articles: List[Dict], 
        items: List[Dict]
    ) -> List[Dict[str, Any]]:
        """조례문서 계층 구조 구성"""
        hierarchy = []
        
        # 간단한 계층 구조 (실제로는 더 복잡한 로직 필요)
        current_chapter = None
        current_section = None
        current_article = None
        
        all_elements = []
        
        # 모든 요소를 위치순으로 정렬
        for chapter in chapters:
            all_elements.append(("chapter", chapter))
        for section in sections:
            all_elements.append(("section", section))
        for article in articles:
            all_elements.append(("article", article))
        for item in items:
            all_elements.append(("item", item))
        
        all_elements.sort(key=lambda x: x[1]["position"][0])
        
        # 계층 구조 구성
        for element_type, element in all_elements:
            if element_type == "chapter":
                current_chapter = {
                    "type": "chapter",
                    "content": element,
                    "children": []
                }
                hierarchy.append(current_chapter)
                current_section = None
                current_article = None
                
            elif element_type == "section":
                section_node = {
                    "type": "section",
                    "content": element,
                    "children": []
                }
                
                if current_chapter:
                    current_chapter["children"].append(section_node)
                else:
                    hierarchy.append(section_node)
                
                current_section = section_node
                current_article = None
                
            elif element_type == "article":
                article_node = {
                    "type": "article",
                    "content": element,
                    "children": []
                }
                
                if current_section:
                    current_section["children"].append(article_node)
                elif current_chapter:
                    current_chapter["children"].append(article_node)
                else:
                    hierarchy.append(article_node)
                
                current_article = article_node
                
            elif element_type == "item":
                item_node = {
                    "type": "item",
                    "content": element,
                    "children": []
                }
                
                if current_article:
                    current_article["children"].append(item_node)
                elif current_section:
                    current_section["children"].append(item_node)
                elif current_chapter:
                    current_chapter["children"].append(item_node)
                else:
                    hierarchy.append(item_node)
        
        return hierarchy
    
    def _build_administrative_hierarchy(
        self, 
        sections: List[Dict], 
        subsections: List[Dict], 
        items: List[Dict]
    ) -> List[Dict[str, Any]]:
        """행정문서 계층 구조 구성"""
        # 간단한 구현 (실제로는 더 정교한 로직 필요)
        hierarchy = []
        
        for section in sections:
            section_node = {
                "type": "section",
                "content": section,
                "children": []
            }
            hierarchy.append(section_node)
        
        return hierarchy
    
    def _analyze_civil_affairs_patterns(self, content: str) -> Dict[str, Any]:
        """민원문서 특별 패턴 분석"""
        patterns = {}
        
        # 처리기간 패턴
        period_pattern = re.compile(r'처리기간\s*[:：]\s*([^\n]+)')
        period_matches = period_pattern.findall(content)
        patterns["processing_periods"] = period_matches
        
        # 수수료 패턴
        fee_pattern = re.compile(r'수수료\s*[:：]\s*([^\n]+)')
        fee_matches = fee_pattern.findall(content)
        patterns["fees"] = fee_matches
        
        # 담당부서 패턴
        dept_pattern = re.compile(r'담당부서\s*[:：]\s*([^\n]+)')
        dept_matches = dept_pattern.findall(content)
        patterns["departments"] = dept_matches
        
        return patterns
    
    async def cleanup(self):
        """리소스 정리"""
        self.compiled_patterns = {}
        self.is_initialized = False
        logger.info("구조 분석기 정리 완료")