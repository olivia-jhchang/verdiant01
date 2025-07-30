"""
데이터베이스 연결 관리
"""
import sqlite3
import time
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from shared.models.config import DatabaseConfig
from shared.models.exceptions import DatabaseConnectionError
from shared.utils.logging import setup_logging

logger = setup_logging("database")


class DatabaseManager:
    """데이터베이스 연결 관리자"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = []
        self.max_retries = 3
        self.retry_delay = 1
        
        # 로컬 개발용 SQLite 데이터베이스 초기화
        if config.driver == "sqlite":
            self._init_sqlite_db()
    
    def _init_sqlite_db(self):
        """SQLite 데이터베이스 초기화 (로컬 개발용)"""
        try:
            conn = sqlite3.connect("intelligent_search.db")
            cursor = conn.cursor()
            
            # 샘플 문서 테이블 생성
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            # 샘플 데이터 삽입 (테스트용)
            sample_documents = [
                ("민원 처리 절차 안내", "민원 접수부터 처리 완료까지의 전체 절차를 안내합니다. 1. 민원 접수: 온라인 또는 방문 접수 가능 2. 접수 확인: 접수증 발급 3. 담당부서 배정: 업무 성격에 따라 담당부서 배정 4. 검토 및 처리: 관련 법령 검토 후 처리 5. 결과 통보: 처리 결과를 신청인에게 통보", "민원문서"),
                ("건축허가 신청 안내", "건축허가 신청 시 필요한 서류와 절차를 안내합니다. 필요서류: 1. 건축허가신청서 2. 설계도서 3. 토지이용계획확인서 4. 건축물 배치도 처리기간: 일반건축물 14일, 복잡한 건축물 21일", "민원문서"),
                ("제1조 목적", "이 조례는 시민의 편의를 도모하고 행정서비스의 질을 향상시키기 위하여 필요한 사항을 규정함을 목적으로 한다.", "조례문서"),
                ("제2조 정의", "이 조례에서 사용하는 용어의 뜻은 다음과 같다. 1. '민원'이란 시민이 행정기관에 대하여 처분 등 특정한 행위를 요구하는 것을 말한다. 2. '처리기간'이란 민원을 접수한 날부터 처리가 완료되는 날까지의 기간을 말한다.", "조례문서"),
                ("예산 집행 지침", "2024년도 예산 집행에 관한 지침을 다음과 같이 시행한다. 1. 예산 집행 원칙: 계획성, 효율성, 투명성을 기본으로 한다. 2. 집행 절차: 집행계획 수립 → 집행 승인 → 집행 → 정산 3. 주의사항: 목적 외 사용 금지, 증빙서류 보관", "행정문서"),
                ("개인정보 처리 방침", "개인정보보호법에 따른 개인정보 처리 방침을 다음과 같이 정한다. 1. 개인정보 수집 목적: 민원 처리, 행정서비스 제공 2. 수집 항목: 성명, 주소, 연락처 3. 보유기간: 처리 완료 후 3년 4. 제3자 제공: 법령에 의한 경우를 제외하고 제공하지 않음", "행정문서")
            ]
            
            # 기존 데이터 확인
            cursor.execute("SELECT COUNT(*) FROM documents")
            count = cursor.fetchone()[0]
            
            if count == 0:
                cursor.executemany(
                    "INSERT INTO documents (title, content, document_type) VALUES (?, ?, ?)",
                    sample_documents
                )
                logger.info(f"샘플 문서 {len(sample_documents)}건을 삽입했습니다.")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"SQLite 데이터베이스 초기화 실패: {e}")
            raise DatabaseConnectionError(f"데이터베이스 초기화 실패: {e}")
    
    @contextmanager
    def get_connection(self):
        """데이터베이스 연결 컨텍스트 매니저"""
        connection = None
        try:
            connection = self._create_connection()
            yield connection
        except Exception as e:
            if connection:
                connection.rollback()
            raise e
        finally:
            if connection:
                connection.close()
    
    def _create_connection(self):
        """데이터베이스 연결 생성"""
        for attempt in range(self.max_retries):
            try:
                if self.config.driver == "sqlite":
                    connection = sqlite3.connect(self.config.database + ".db")
                    connection.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
                    return connection
                else:
                    # 실제 환경에서는 PostgreSQL, MySQL 등 연결
                    raise NotImplementedError("SQLite 외의 데이터베이스는 아직 구현되지 않았습니다.")
                
            except Exception as e:
                logger.warning(f"데이터베이스 연결 시도 {attempt + 1} 실패: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))  # 지수 백오프
                else:
                    raise DatabaseConnectionError(f"데이터베이스 연결 실패: {e}")
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """쿼리 실행 (SELECT)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 결과를 딕셔너리 리스트로 변환
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """쿼리 실행 (INSERT, UPDATE, DELETE)"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            conn.commit()
            return cursor.rowcount
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """테이블 스키마 조회"""
        if self.config.driver == "sqlite":
            query = f"PRAGMA table_info({table_name})"
        else:
            # 다른 데이터베이스의 경우 적절한 쿼리 사용
            raise NotImplementedError("SQLite 외의 데이터베이스는 아직 구현되지 않았습니다.")
        
        return self.execute_query(query)
    
    def get_table_list(self) -> List[str]:
        """테이블 목록 조회"""
        if self.config.driver == "sqlite":
            query = "SELECT name FROM sqlite_master WHERE type='table'"
        else:
            raise NotImplementedError("SQLite 외의 데이터베이스는 아직 구현되지 않았습니다.")
        
        results = self.execute_query(query)
        return [row['name'] for row in results]