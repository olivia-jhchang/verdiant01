"""
문서 추출기
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from shared.models.document import Document, DocumentType
from shared.utils.logging import setup_logging, log_with_context
from shared.utils.korean_utils import detect_document_structure, clean_korean_text

logger = setup_logging("document-extractor")


class DocumentExtractor:
    """문서 추출 및 변환 클래스"""
    
    def __init__(self):
        self.extraction_stats = {
            "total_extracted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "type_distribution": {
                "행정문서": 0,
                "민원문서": 0,
                "조례문서": 0
            }
        }
    
    def extract_and_transform_documents(
        self, 
        raw_data: List[Dict[str, Any]], 
        table_name: str,
        column_mapping: Dict[str, str]
    ) -> List[Document]:
        """원시 데이터를 Document 객체로 변환"""
        documents = []
        
        log_with_context(
            logger, "info",
            "문서 추출 시작",
            table_name=table_name,
            raw_data_count=len(raw_data),
            column_mapping=column_mapping
        )
        
        for i, row in enumerate(raw_data):
            try:
                document = self._transform_row_to_document(
                    row, table_name, column_mapping
                )
                documents.append(document)
                self.extraction_stats["successful_extractions"] += 1
                self.extraction_stats["type_distribution"][document.document_type.value] += 1
                
            except Exception as e:
                self.extraction_stats["failed_extractions"] += 1
                log_with_context(
                    logger, "warning",
                    "문서 변환 실패",
                    row_index=i,
                    table_name=table_name,
                    error=str(e)
                )
                continue
        
        self.extraction_stats["total_extracted"] += len(documents)
        
        log_with_context(
            logger, "info",
            "문서 추출 완료",
            table_name=table_name,
            extracted_count=len(documents),
            stats=self.extraction_stats
        )
        
        return documents
    
    def _transform_row_to_document(
        self, 
        row: Dict[str, Any], 
        table_name: str,
        column_mapping: Dict[str, str]
    ) -> Document:
        """단일 행을 Document 객체로 변환"""
        # 기본 필드 추출
        doc_id = self._extract_field(row, column_mapping, 'id', str(row.get('id', '')))
        title = self._extract_field(row, column_mapping, 'title', f"문서 {doc_id}")
        content = self._extract_field(row, column_mapping, 'content', '')
        
        # 텍스트 정리
        title = clean_korean_text(title)
        content = clean_korean_text(content)
        
        # 문서 타입 결정
        document_type = self._determine_document_type(
            row, column_mapping, title, content
        )
        
        # 날짜 처리
        created_at, updated_at = self._extract_dates(row, column_mapping)
        
        # 메타데이터 구성
        metadata = self._build_metadata(row, column_mapping, table_name)
        
        # 문서 구조 분석 추가
        structure_info = detect_document_structure(content)
        metadata['structure'] = structure_info
        
        return Document(
            id=doc_id,
            title=title,
            content=content,
            document_type=document_type,
            source_table=table_name,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata
        )
    
    def _extract_field(
        self, 
        row: Dict[str, Any], 
        column_mapping: Dict[str, str],
        field_name: str,
        default_value: Any = None
    ) -> Any:
        """매핑된 컬럼에서 필드 값 추출"""
        column_name = column_mapping.get(field_name)
        if column_name and column_name in row:
            value = row[column_name]
            return value if value is not None else default_value
        return default_value
    
    def _determine_document_type(
        self, 
        row: Dict[str, Any], 
        column_mapping: Dict[str, str],
        title: str,
        content: str
    ) -> DocumentType:
        """문서 타입 결정"""
        # 명시적 문서 타입이 있는 경우
        doc_type_value = self._extract_field(row, column_mapping, 'document_type')
        if doc_type_value:
            type_mapping = {
                '행정문서': DocumentType.ADMINISTRATIVE,
                '민원문서': DocumentType.CIVIL_AFFAIRS,
                '조례문서': DocumentType.REGULATION,
                'administrative': DocumentType.ADMINISTRATIVE,
                'civil': DocumentType.CIVIL_AFFAIRS,
                'regulation': DocumentType.REGULATION,
                'admin': DocumentType.ADMINISTRATIVE,
                'civil_affairs': DocumentType.CIVIL_AFFAIRS
            }
            
            mapped_type = type_mapping.get(str(doc_type_value).lower())
            if mapped_type:
                return mapped_type
        
        # 내용 기반 타입 추정
        return self._infer_document_type_from_content(title, content)
    
    def _infer_document_type_from_content(self, title: str, content: str) -> DocumentType:
        """내용 기반 문서 타입 추정"""
        text = (title + " " + content).lower()
        
        # 조례문서 패턴 (가장 구체적인 패턴부터)
        regulation_patterns = [
            '제1조', '제2조', '제3조',  # 조항 번호
            '조례', '규정', '시행령',
            '법령', '규칙', '고시',
            '제.*조.*목적', '제.*조.*정의'
        ]
        
        regulation_score = sum(1 for pattern in regulation_patterns if pattern in text)
        
        # 민원문서 패턴
        civil_patterns = [
            '민원', '신청', '접수', '처리',
            '허가', '승인', '인허가',
            '신청서', '접수증', '처리결과',
            '건축허가', '사업허가', '영업허가'
        ]
        
        civil_score = sum(1 for pattern in civil_patterns if pattern in text)
        
        # 행정문서 패턴
        admin_patterns = [
            '공문', '시행', '시달', '통보',
            '지시', '업무', '처리', '보고',
            '계획', '예산', '집행', '결재'
        ]
        
        admin_score = sum(1 for pattern in admin_patterns if pattern in text)
        
        # 점수 기반 분류
        scores = {
            DocumentType.REGULATION: regulation_score,
            DocumentType.CIVIL_AFFAIRS: civil_score,
            DocumentType.ADMINISTRATIVE: admin_score
        }
        
        # 가장 높은 점수의 타입 반환
        max_type = max(scores, key=scores.get)
        
        # 모든 점수가 0이면 기본값 (행정문서)
        if scores[max_type] == 0:
            return DocumentType.ADMINISTRATIVE
        
        return max_type
    
    def _extract_dates(
        self, 
        row: Dict[str, Any], 
        column_mapping: Dict[str, str]
    ) -> Tuple[datetime, datetime]:
        """생성일/수정일 추출"""
        now = datetime.now()
        
        # 생성일 추출
        created_at = self._extract_field(row, column_mapping, 'created_at')
        if created_at:
            created_at = self._parse_datetime(created_at, now)
        else:
            created_at = now
        
        # 수정일 추출
        updated_at = self._extract_field(row, column_mapping, 'updated_at')
        if updated_at:
            updated_at = self._parse_datetime(updated_at, created_at)
        else:
            updated_at = created_at
        
        return created_at, updated_at
    
    def _parse_datetime(self, date_value: Any, default: datetime) -> datetime:
        """다양한 형식의 날짜를 datetime으로 변환"""
        if isinstance(date_value, datetime):
            return date_value
        
        if isinstance(date_value, str):
            # 다양한 날짜 형식 시도
            date_formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d',
                '%Y/%m/%d',
                '%Y.%m.%d'
            ]
            
            for fmt in date_formats:
                try:
                    return datetime.strptime(date_value, fmt)
                except ValueError:
                    continue
            
            # ISO 형식 시도
            try:
                return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except ValueError:
                pass
        
        # 변환 실패 시 기본값 반환
        return default
    
    def _build_metadata(
        self, 
        row: Dict[str, Any], 
        column_mapping: Dict[str, str],
        table_name: str
    ) -> Dict[str, Any]:
        """메타데이터 구성"""
        metadata = {
            'source_table': table_name,
            'extraction_time': datetime.now().isoformat(),
            'column_mapping': column_mapping,
            'original_data': {}
        }
        
        # 매핑되지 않은 추가 컬럼들을 메타데이터에 포함
        mapped_columns = set(column_mapping.values())
        for key, value in row.items():
            if key not in mapped_columns:
                # JSON 직렬화 가능한 값만 포함
                try:
                    json.dumps(value)
                    metadata['original_data'][key] = value
                except (TypeError, ValueError):
                    metadata['original_data'][key] = str(value)
        
        return metadata
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """추출 통계 반환"""
        return self.extraction_stats.copy()
    
    def reset_stats(self):
        """통계 초기화"""
        self.extraction_stats = {
            "total_extracted": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "type_distribution": {
                "행정문서": 0,
                "민원문서": 0,
                "조례문서": 0
            }
        }