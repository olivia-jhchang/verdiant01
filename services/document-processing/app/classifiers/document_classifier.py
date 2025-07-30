"""
문서 분류기
"""
import re
from typing import Dict, List, Tuple
from shared.models.document import DocumentType
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("document-classifier")


class DocumentClassifier:
    """문서 타입 분류기"""
    
    def __init__(self):
        self.is_initialized = False
        
        # 문서 타입별 키워드 패턴
        self.classification_patterns = {
            DocumentType.REGULATION: {
                "keywords": [
                    "조례", "규정", "시행령", "법령", "규칙", "고시",
                    "제1조", "제2조", "제3조", "조항", "항목",
                    "목적", "정의", "적용범위", "시행일"
                ],
                "patterns": [
                    r'제\s*\d+\s*조',  # 제1조, 제2조
                    r'제\s*[일이삼사오육칠팔구십]+\s*조',  # 제일조, 제이조
                    r'제\s*\d+\s*장',  # 제1장
                    r'제\s*\d+\s*절',  # 제1절
                    r'조례|규정|시행령|법령'
                ],
                "weight": 1.0
            },
            DocumentType.CIVIL_AFFAIRS: {
                "keywords": [
                    "민원", "신청", "접수", "처리", "허가", "승인", "인허가",
                    "신청서", "접수증", "처리결과", "승인서", "허가서",
                    "건축허가", "사업허가", "영업허가", "개발허가",
                    "필요서류", "구비서류", "제출서류", "처리기간",
                    "수수료", "담당부서", "문의처"
                ],
                "patterns": [
                    r'민원\s*(처리|접수|신청)',
                    r'(건축|사업|영업|개발)\s*허가',
                    r'신청서|접수증|처리결과',
                    r'필요서류|구비서류|제출서류',
                    r'처리기간\s*[:：]',
                    r'수수료\s*[:：]'
                ],
                "weight": 1.0
            },
            DocumentType.ADMINISTRATIVE: {
                "keywords": [
                    "공문", "시행", "시달", "통보", "지시", "업무", "처리",
                    "보고", "계획", "예산", "집행", "결재", "승인",
                    "회의", "협의", "검토", "조치", "이행", "추진",
                    "부서", "담당자", "일정", "현황", "실적"
                ],
                "patterns": [
                    r'공문\s*(시행|시달|통보)',
                    r'업무\s*(처리|보고|계획)',
                    r'예산\s*(집행|편성|배정)',
                    r'회의\s*(개최|결과|안건)',
                    r'검토\s*(결과|의견|사항)'
                ],
                "weight": 1.0
            }
        }
        
        # 제외 키워드 (분류 정확도 향상)
        self.exclusion_patterns = {
            DocumentType.REGULATION: ["민원처리", "신청방법"],
            DocumentType.CIVIL_AFFAIRS: ["조례제정", "규정개정"],
            DocumentType.ADMINISTRATIVE: ["민원신청", "허가절차"]
        }
    
    async def initialize(self):
        """분류기 초기화"""
        try:
            # 패턴 컴파일
            self.compiled_patterns = {}
            for doc_type, type_info in self.classification_patterns.items():
                self.compiled_patterns[doc_type] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in type_info["patterns"]
                ]
            
            # 제외 패턴 컴파일
            self.compiled_exclusions = {}
            for doc_type, exclusions in self.exclusion_patterns.items():
                self.compiled_exclusions[doc_type] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in exclusions
                ]
            
            self.is_initialized = True
            logger.info("문서 분류기 초기화 완료")
            
        except Exception as e:
            logger.error(f"문서 분류기 초기화 실패: {e}")
            raise e
    
    async def classify_document(self, title: str, content: str) -> DocumentType:
        """문서 분류"""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # 제목과 내용 결합
            full_text = f"{title} {content}".lower()
            
            # 각 문서 타입별 점수 계산
            scores = {}
            for doc_type in DocumentType:
                score = await self._calculate_type_score(full_text, doc_type)
                scores[doc_type] = score
            
            # 가장 높은 점수의 타입 선택
            best_type = max(scores, key=scores.get)
            best_score = scores[best_type]
            
            log_with_context(
                logger, "debug",
                "문서 분류 완료",
                scores=scores,
                best_type=best_type.value,
                best_score=best_score
            )
            
            # 최소 임계값 확인
            if best_score < 0.3:
                return DocumentType.ADMINISTRATIVE  # 기본값
            
            return best_type
            
        except Exception as e:
            logger.warning(f"문서 분류 실패: {e}")
            return DocumentType.ADMINISTRATIVE  # 기본값
    
    async def get_classification_confidence(
        self, 
        title: str, 
        content: str, 
        predicted_type: DocumentType
    ) -> float:
        """분류 신뢰도 계산"""
        try:
            full_text = f"{title} {content}".lower()
            
            # 예측된 타입의 점수 계산
            type_score = await self._calculate_type_score(full_text, predicted_type)
            
            # 다른 타입들의 점수 계산
            other_scores = []
            for doc_type in DocumentType:
                if doc_type != predicted_type:
                    score = await self._calculate_type_score(full_text, doc_type)
                    other_scores.append(score)
            
            # 신뢰도 계산 (예측 타입 점수 vs 다른 타입들의 최대 점수)
            max_other_score = max(other_scores) if other_scores else 0
            
            if type_score + max_other_score == 0:
                return 0.5  # 중립
            
            confidence = type_score / (type_score + max_other_score)
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.warning(f"신뢰도 계산 실패: {e}")
            return 0.5
    
    async def _calculate_type_score(self, text: str, doc_type: DocumentType) -> float:
        """특정 문서 타입에 대한 점수 계산"""
        if doc_type not in self.classification_patterns:
            return 0.0
        
        type_info = self.classification_patterns[doc_type]
        total_score = 0.0
        
        # 키워드 점수
        keyword_score = self._calculate_keyword_score(text, type_info["keywords"])
        
        # 패턴 점수
        pattern_score = self._calculate_pattern_score(text, doc_type)
        
        # 제외 패턴 페널티
        exclusion_penalty = self._calculate_exclusion_penalty(text, doc_type)
        
        # 구조적 특징 점수
        structural_score = self._calculate_structural_score(text, doc_type)
        
        # 총 점수 계산
        total_score = (
            keyword_score * 0.4 +
            pattern_score * 0.3 +
            structural_score * 0.2 +
            exclusion_penalty * 0.1
        ) * type_info["weight"]
        
        return max(0.0, total_score)
    
    def _calculate_keyword_score(self, text: str, keywords: List[str]) -> float:
        """키워드 기반 점수 계산"""
        if not keywords:
            return 0.0
        
        found_keywords = 0
        total_occurrences = 0
        
        for keyword in keywords:
            occurrences = len(re.findall(re.escape(keyword), text, re.IGNORECASE))
            if occurrences > 0:
                found_keywords += 1
                total_occurrences += occurrences
        
        # 키워드 다양성 점수 (0-1)
        diversity_score = found_keywords / len(keywords)
        
        # 키워드 빈도 점수 (정규화)
        frequency_score = min(total_occurrences / 10, 1.0)
        
        return (diversity_score + frequency_score) / 2
    
    def _calculate_pattern_score(self, text: str, doc_type: DocumentType) -> float:
        """패턴 기반 점수 계산"""
        if doc_type not in self.compiled_patterns:
            return 0.0
        
        patterns = self.compiled_patterns[doc_type]
        if not patterns:
            return 0.0
        
        found_patterns = 0
        total_matches = 0
        
        for pattern in patterns:
            matches = len(pattern.findall(text))
            if matches > 0:
                found_patterns += 1
                total_matches += matches
        
        # 패턴 다양성 점수
        diversity_score = found_patterns / len(patterns)
        
        # 패턴 빈도 점수
        frequency_score = min(total_matches / 5, 1.0)
        
        return (diversity_score + frequency_score) / 2
    
    def _calculate_exclusion_penalty(self, text: str, doc_type: DocumentType) -> float:
        """제외 패턴 페널티 계산"""
        if doc_type not in self.compiled_exclusions:
            return 0.0
        
        exclusion_patterns = self.compiled_exclusions[doc_type]
        if not exclusion_patterns:
            return 0.0
        
        penalty = 0.0
        for pattern in exclusion_patterns:
            matches = len(pattern.findall(text))
            penalty += matches * 0.1  # 각 매치당 0.1 페널티
        
        return -min(penalty, 0.5)  # 최대 0.5 페널티
    
    def _calculate_structural_score(self, text: str, doc_type: DocumentType) -> float:
        """구조적 특징 점수 계산"""
        if doc_type == DocumentType.REGULATION:
            # 조례문서는 조항 구조가 중요
            article_pattern = re.compile(r'제\s*\d+\s*조', re.IGNORECASE)
            article_count = len(article_pattern.findall(text))
            
            if article_count >= 3:
                return 1.0
            elif article_count >= 1:
                return 0.7
            else:
                return 0.0
        
        elif doc_type == DocumentType.CIVIL_AFFAIRS:
            # 민원문서는 절차나 서류 목록이 중요
            procedure_patterns = [
                r'\d+\s*단계',  # 1단계, 2단계
                r'\d+\)\s*',    # 1) 2) 3)
                r'[-•]\s*',     # - 또는 •
            ]
            
            total_matches = 0
            for pattern_str in procedure_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                total_matches += len(pattern.findall(text))
            
            if total_matches >= 5:
                return 1.0
            elif total_matches >= 2:
                return 0.6
            else:
                return 0.0
        
        elif doc_type == DocumentType.ADMINISTRATIVE:
            # 행정문서는 공문 형식이나 업무 관련 구조가 중요
            admin_patterns = [
                r'\d+\.\s*',    # 1. 2. 3.
                r'([IVX]+)\.\s*',  # I. II. III.
                r'([가나다라마바사아자차카타파하])\)\s*',  # 가) 나) 다)
            ]
            
            total_matches = 0
            for pattern_str in admin_patterns:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                total_matches += len(pattern.findall(text))
            
            if total_matches >= 3:
                return 0.8
            elif total_matches >= 1:
                return 0.5
            else:
                return 0.0
        
        return 0.0
    
    async def get_classification_details(
        self, 
        title: str, 
        content: str
    ) -> Dict[str, Any]:
        """분류 상세 정보 반환"""
        if not self.is_initialized:
            await self.initialize()
        
        full_text = f"{title} {content}".lower()
        
        details = {
            "scores": {},
            "found_keywords": {},
            "found_patterns": {},
            "structural_features": {}
        }
        
        for doc_type in DocumentType:
            # 점수 계산
            score = await self._calculate_type_score(full_text, doc_type)
            details["scores"][doc_type.value] = score
            
            # 발견된 키워드
            type_info = self.classification_patterns[doc_type]
            found_keywords = []
            for keyword in type_info["keywords"]:
                if keyword in full_text:
                    found_keywords.append(keyword)
            details["found_keywords"][doc_type.value] = found_keywords
            
            # 발견된 패턴
            found_patterns = []
            if doc_type in self.compiled_patterns:
                for pattern in self.compiled_patterns[doc_type]:
                    matches = pattern.findall(full_text)
                    if matches:
                        found_patterns.extend(matches)
            details["found_patterns"][doc_type.value] = found_patterns
        
        return details
    
    async def cleanup(self):
        """리소스 정리"""
        self.compiled_patterns = {}
        self.compiled_exclusions = {}
        self.is_initialized = False
        logger.info("문서 분류기 정리 완료")