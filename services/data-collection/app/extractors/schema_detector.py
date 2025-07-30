"""
데이터베이스 스키마 감지 및 분석
"""
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("schema-detector")


class SchemaDetector:
    """데이터베이스 스키마 감지 및 분석 클래스"""
    
    def __init__(self):
        self.schema_cache = {}
        self.column_patterns = {
            'id': ['id', 'document_id', 'doc_id', 'seq', 'no', 'num'],
            'title': ['title', 'subject', 'name', 'heading', 'caption'],
            'content': ['content', 'text', 'body', 'description', 'detail', 'memo'],
            'document_type': ['document_type', 'type', 'category', 'doc_type', 'kind'],
            'created_at': ['created_at', 'created', 'date_created', 'insert_date', 'reg_date'],
            'updated_at': ['updated_at', 'updated', 'date_updated', 'modify_date', 'upd_date']
        }
    
    def analyze_table_schema(
        self, 
        table_name: str, 
        schema_info: List[Dict[str, Any]],
        sample_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """테이블 스키마 분석"""
        log_with_context(
            logger, "info",
            "테이블 스키마 분석 시작",
            table_name=table_name,
            column_count=len(schema_info)
        )
        
        analysis = {
            'table_name': table_name,
            'column_count': len(schema_info),
            'columns': [],
            'column_mapping': {},
            'data_quality': {},
            'recommendations': [],
            'is_document_table': False,
            'confidence_score': 0.0
        }
        
        # 컬럼별 분석
        for col_info in schema_info:
            column_analysis = self._analyze_column(col_info, sample_data)
            analysis['columns'].append(column_analysis)
        
        # 컬럼 매핑 결정
        analysis['column_mapping'] = self._determine_optimal_mapping(
            analysis['columns']
        )
        
        # 문서 테이블 여부 판단
        analysis['is_document_table'] = self._is_document_table(
            analysis['column_mapping'], analysis['columns']
        )
        
        # 신뢰도 점수 계산
        analysis['confidence_score'] = self._calculate_confidence_score(
            analysis['column_mapping'], analysis['columns']
        )
        
        # 데이터 품질 분석
        if sample_data:
            analysis['data_quality'] = self._analyze_data_quality(
                sample_data, analysis['column_mapping']
            )
        
        # 권장사항 생성
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # 캐시에 저장
        self.schema_cache[table_name] = analysis
        
        log_with_context(
            logger, "info",
            "테이블 스키마 분석 완료",
            table_name=table_name,
            is_document_table=analysis['is_document_table'],
            confidence_score=analysis['confidence_score']
        )
        
        return analysis
    
    def _analyze_column(
        self, 
        col_info: Dict[str, Any], 
        sample_data: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """개별 컬럼 분석"""
        column_name = col_info.get('name', '').lower()
        column_type = col_info.get('type', '').lower()
        
        analysis = {
            'name': col_info.get('name', ''),
            'type': column_type,
            'nullable': col_info.get('notnull', 0) == 0,
            'primary_key': col_info.get('pk', 0) == 1,
            'purpose': 'unknown',
            'confidence': 0.0,
            'sample_values': [],
            'data_characteristics': {}
        }
        
        # 컬럼 목적 추정
        analysis['purpose'], analysis['confidence'] = self._infer_column_purpose(
            column_name, column_type
        )
        
        # 샘플 데이터 분석
        if sample_data:
            analysis['sample_values'] = self._extract_sample_values(
                sample_data, analysis['name']
            )
            analysis['data_characteristics'] = self._analyze_column_data(
                analysis['sample_values'], column_type
            )
        
        return analysis
    
    def _infer_column_purpose(self, column_name: str, column_type: str) -> Tuple[str, float]:
        """컬럼 목적 추정"""
        column_name = column_name.lower()
        
        # 정확한 매칭 (높은 신뢰도)
        for purpose, patterns in self.column_patterns.items():
            if column_name in patterns:
                return purpose, 0.9
        
        # 부분 매칭 (중간 신뢰도)
        for purpose, patterns in self.column_patterns.items():
            for pattern in patterns:
                if pattern in column_name or column_name in pattern:
                    return purpose, 0.7
        
        # 타입 기반 추정 (낮은 신뢰도)
        if 'int' in column_type or 'serial' in column_type:
            if 'id' in column_name or column_name.endswith('_id'):
                return 'id', 0.5
        
        if 'text' in column_type or 'varchar' in column_type or 'char' in column_type:
            if len(column_name) > 10:  # 긴 컬럼명은 내용일 가능성
                return 'content', 0.3
            else:
                return 'title', 0.3
        
        if 'date' in column_type or 'time' in column_type:
            if 'creat' in column_name or 'insert' in column_name:
                return 'created_at', 0.6
            elif 'updat' in column_name or 'modif' in column_name:
                return 'updated_at', 0.6
            else:
                return 'created_at', 0.4
        
        return 'unknown', 0.0
    
    def _extract_sample_values(
        self, 
        sample_data: List[Dict[str, Any]], 
        column_name: str
    ) -> List[Any]:
        """샘플 값 추출"""
        values = []
        for row in sample_data[:10]:  # 최대 10개 샘플
            if column_name in row and row[column_name] is not None:
                values.append(row[column_name])
        return values
    
    def _analyze_column_data(self, sample_values: List[Any], column_type: str) -> Dict[str, Any]:
        """컬럼 데이터 특성 분석"""
        if not sample_values:
            return {'empty': True}
        
        characteristics = {
            'empty': False,
            'sample_count': len(sample_values),
            'unique_count': len(set(str(v) for v in sample_values)),
            'avg_length': 0,
            'has_korean': False,
            'has_numbers': False,
            'has_dates': False
        }
        
        # 문자열 분석
        if sample_values and isinstance(sample_values[0], str):
            total_length = sum(len(str(v)) for v in sample_values)
            characteristics['avg_length'] = total_length / len(sample_values)
            
            # 한국어 포함 여부
            korean_pattern = re.compile(r'[가-힣]')
            characteristics['has_korean'] = any(
                korean_pattern.search(str(v)) for v in sample_values
            )
            
            # 숫자 포함 여부
            characteristics['has_numbers'] = any(
                re.search(r'\d', str(v)) for v in sample_values
            )
            
            # 날짜 패턴 여부
            date_pattern = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}')
            characteristics['has_dates'] = any(
                date_pattern.search(str(v)) for v in sample_values
            )
        
        return characteristics
    
    def _determine_optimal_mapping(self, columns: List[Dict[str, Any]]) -> Dict[str, str]:
        """최적 컬럼 매핑 결정"""
        mapping = {}
        
        # 각 목적별로 가장 신뢰도 높은 컬럼 선택
        purpose_candidates = {}
        
        for col in columns:
            purpose = col['purpose']
            confidence = col['confidence']
            
            if purpose != 'unknown':
                if purpose not in purpose_candidates or confidence > purpose_candidates[purpose]['confidence']:
                    purpose_candidates[purpose] = {
                        'column_name': col['name'],
                        'confidence': confidence
                    }
        
        # 매핑 생성
        for purpose, candidate in purpose_candidates.items():
            if candidate['confidence'] > 0.3:  # 최소 신뢰도 임계값
                mapping[purpose] = candidate['column_name']
        
        return mapping
    
    def _is_document_table(
        self, 
        column_mapping: Dict[str, str], 
        columns: List[Dict[str, Any]]
    ) -> bool:
        """문서 테이블 여부 판단"""
        # 필수 컬럼 확인
        required_fields = ['id', 'content']
        has_required = all(field in column_mapping for field in required_fields)
        
        if not has_required:
            return False
        
        # 내용 컬럼의 특성 확인
        content_column = column_mapping.get('content')
        if content_column:
            content_col_info = next(
                (col for col in columns if col['name'] == content_column), 
                None
            )
            
            if content_col_info:
                characteristics = content_col_info.get('data_characteristics', {})
                # 평균 길이가 충분히 길고 한국어를 포함하는지 확인
                if (characteristics.get('avg_length', 0) > 50 and 
                    characteristics.get('has_korean', False)):
                    return True
        
        return False
    
    def _calculate_confidence_score(
        self, 
        column_mapping: Dict[str, str], 
        columns: List[Dict[str, Any]]
    ) -> float:
        """전체 신뢰도 점수 계산"""
        if not column_mapping:
            return 0.0
        
        total_confidence = 0.0
        weight_sum = 0.0
        
        # 각 필드별 가중치
        field_weights = {
            'id': 1.0,
            'title': 0.8,
            'content': 1.5,  # 가장 중요
            'document_type': 0.6,
            'created_at': 0.4,
            'updated_at': 0.3
        }
        
        for field, column_name in column_mapping.items():
            weight = field_weights.get(field, 0.5)
            
            # 해당 컬럼의 신뢰도 찾기
            col_info = next(
                (col for col in columns if col['name'] == column_name), 
                None
            )
            
            if col_info:
                confidence = col_info.get('confidence', 0.0)
                total_confidence += confidence * weight
                weight_sum += weight
        
        return total_confidence / weight_sum if weight_sum > 0 else 0.0
    
    def _analyze_data_quality(
        self, 
        sample_data: List[Dict[str, Any]], 
        column_mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        """데이터 품질 분석"""
        quality = {
            'sample_size': len(sample_data),
            'completeness': {},
            'consistency': {},
            'issues': []
        }
        
        if not sample_data:
            return quality
        
        # 완전성 분석 (NULL 값 비율)
        for field, column_name in column_mapping.items():
            non_null_count = sum(
                1 for row in sample_data 
                if row.get(column_name) is not None and str(row.get(column_name)).strip()
            )
            completeness = non_null_count / len(sample_data)
            quality['completeness'][field] = completeness
            
            if completeness < 0.8:
                quality['issues'].append(f"{field} 필드의 완전성이 낮습니다 ({completeness:.1%})")
        
        # 일관성 분석
        content_column = column_mapping.get('content')
        if content_column:
            content_lengths = [
                len(str(row.get(content_column, ''))) 
                for row in sample_data 
                if row.get(content_column)
            ]
            
            if content_lengths:
                avg_length = sum(content_lengths) / len(content_lengths)
                quality['consistency']['avg_content_length'] = avg_length
                
                if avg_length < 20:
                    quality['issues'].append("내용 필드의 평균 길이가 너무 짧습니다")
        
        return quality
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []
        
        # 문서 테이블이 아닌 경우
        if not analysis['is_document_table']:
            recommendations.append("이 테이블은 문서 데이터에 적합하지 않을 수 있습니다")
        
        # 신뢰도가 낮은 경우
        if analysis['confidence_score'] < 0.5:
            recommendations.append("컬럼 매핑의 신뢰도가 낮습니다. 수동 검토가 필요합니다")
        
        # 필수 필드 누락
        required_fields = ['id', 'content']
        missing_fields = [
            field for field in required_fields 
            if field not in analysis['column_mapping']
        ]
        
        if missing_fields:
            recommendations.append(f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}")
        
        # 데이터 품질 이슈
        data_quality = analysis.get('data_quality', {})
        if data_quality.get('issues'):
            recommendations.extend(data_quality['issues'])
        
        return recommendations
    
    def get_cached_analysis(self, table_name: str) -> Optional[Dict[str, Any]]:
        """캐시된 분석 결과 조회"""
        return self.schema_cache.get(table_name)
    
    def clear_cache(self):
        """캐시 초기화"""
        self.schema_cache.clear()