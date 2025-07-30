"""
데이터 수집을 위한 데이터베이스 매니저
"""
import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from shared.database.connection import DatabaseManager
from shared.models.config import DatabaseConfig
from shared.models.document import Document, DocumentType
from shared.models.exceptions import DatabaseConnectionError
from shared.utils.logging import setup_logging, log_with_context
from app.extractors.document_extractor import DocumentExtractor
from app.extractors.schema_detector import SchemaDetector

logger = setup_logging("data-collection-manager")


class DataCollectionManager:
    """데이터 수집 매니저"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.db_manager = DatabaseManager(config)
        self.document_extractor = DocumentExtractor()
        self.schema_detector = SchemaDetector()
        self.connection_status = "disconnected"
        self.last_collection_time = None
        self.collection_stats = {
            "total_documents": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "last_error": None
        }
    
    async def test_connection(self) -> bool:
        """데이터베이스 연결 테스트"""
        try:
            # 비동기 처리를 위해 스레드에서 실행
            loop = asyncio.get_event_loop()
            tables = await loop.run_in_executor(
                None, 
                self.db_manager.get_table_list
            )
            
            self.connection_status = "connected"
            log_with_context(
                logger, "info", 
                "데이터베이스 연결 테스트 성공",
                table_count=len(tables),
                tables=tables
            )
            return True
            
        except Exception as e:
            self.connection_status = "error"
            self.collection_stats["last_error"] = str(e)
            log_with_context(
                logger, "error",
                "데이터베이스 연결 테스트 실패",
                error=str(e)
            )
            raise DatabaseConnectionError(f"연결 테스트 실패: {e}")
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """연결 상태 조회"""
        return {
            "status": self.connection_status,
            "last_collection_time": self.last_collection_time,
            "stats": self.collection_stats,
            "config": {
                "driver": self.config.driver,
                "host": self.config.host,
                "database": self.config.database
            }
        }
    
    async def get_available_tables(self) -> List[Dict[str, Any]]:
        """사용 가능한 테이블 목록 조회"""
        try:
            loop = asyncio.get_event_loop()
            tables = await loop.run_in_executor(
                None,
                self.db_manager.get_table_list
            )
            
            table_info = []
            for table_name in tables:
                try:
                    schema = await loop.run_in_executor(
                        None,
                        self.db_manager.get_table_schema,
                        table_name
                    )
                    
                    # 문서 수 조회
                    count_result = await loop.run_in_executor(
                        None,
                        self.db_manager.execute_query,
                        f"SELECT COUNT(*) as count FROM {table_name}"
                    )
                    
                    document_count = count_result[0]['count'] if count_result else 0
                    
                    # 샘플 데이터 조회 (스키마 분석용)
                    sample_data = []
                    if document_count > 0:
                        sample_result = await loop.run_in_executor(
                            None,
                            self.db_manager.execute_query,
                            f"SELECT * FROM {table_name} LIMIT 5"
                        )
                        sample_data = sample_result
                    
                    # 스키마 분석
                    schema_analysis = self.schema_detector.analyze_table_schema(
                        table_name, schema, sample_data
                    )
                    
                    table_info.append({
                        "table_name": table_name,
                        "document_count": document_count,
                        "schema": schema,
                        "schema_analysis": schema_analysis,
                        "has_text_content": schema_analysis['is_document_table'],
                        "confidence_score": schema_analysis['confidence_score']
                    })
                    
                except Exception as e:
                    logger.warning(f"테이블 {table_name} 정보 조회 실패: {e}")
                    table_info.append({
                        "table_name": table_name,
                        "document_count": 0,
                        "schema": [],
                        "has_text_content": False,
                        "error": str(e)
                    })
            
            return table_info
            
        except Exception as e:
            log_with_context(
                logger, "error",
                "테이블 목록 조회 실패",
                error=str(e)
            )
            raise DatabaseConnectionError(f"테이블 목록 조회 실패: {e}")
    
    def _has_text_content(self, schema: List[Dict[str, Any]]) -> bool:
        """테이블이 텍스트 내용을 포함하는지 확인"""
        text_columns = ['content', 'text', 'body', 'description', 'title']
        column_names = [col.get('name', '').lower() for col in schema]
        
        return any(text_col in column_names for text_col in text_columns)
    
    async def collect_documents(
        self, 
        table_names: List[str] = None,
        incremental: bool = False,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """문서 수집"""
        start_time = time.time()
        collection_result = {
            "success": False,
            "total_documents": 0,
            "processed_documents": 0,
            "failed_documents": 0,
            "errors": [],
            "processing_time": 0,
            "tables_processed": []
        }
        
        try:
            # 테이블 목록이 없으면 모든 테이블 조회
            if not table_names:
                available_tables = await self.get_available_tables()
                table_names = [
                    table['table_name'] 
                    for table in available_tables 
                    if table['has_text_content'] and table['document_count'] > 0
                ]
            
            log_with_context(
                logger, "info",
                "문서 수집 시작",
                tables=table_names,
                incremental=incremental,
                limit=limit
            )
            
            loop = asyncio.get_event_loop()
            
            for table_name in table_names:
                table_result = {
                    "table_name": table_name,
                    "documents_found": 0,
                    "documents_processed": 0,
                    "errors": []
                }
                
                try:
                    # 문서 추출
                    documents = await self._extract_documents_from_table(
                        table_name, incremental, limit
                    )
                    
                    table_result["documents_found"] = len(documents)
                    collection_result["total_documents"] += len(documents)
                    collection_result["processed_documents"] += len(documents)
                    table_result["documents_processed"] = len(documents)
                    
                    # 추출 통계 로깅
                    extractor_stats = self.document_extractor.get_extraction_stats()
                    log_with_context(
                        logger, "info",
                        "테이블 문서 추출 완료",
                        table_name=table_name,
                        documents_extracted=len(documents),
                        extractor_stats=extractor_stats
                    )
                
                except Exception as e:
                    error_msg = f"테이블 {table_name} 처리 실패: {e}"
                    table_result["errors"].append(error_msg)
                    collection_result["errors"].append(error_msg)
                    
                    log_with_context(
                        logger, "error",
                        "테이블 처리 실패",
                        table_name=table_name,
                        error=str(e)
                    )
                
                collection_result["tables_processed"].append(table_result)
            
            # 수집 통계 업데이트
            self.collection_stats["total_documents"] += collection_result["processed_documents"]
            self.collection_stats["successful_collections"] += 1
            self.last_collection_time = datetime.now()
            
            collection_result["success"] = True
            collection_result["processing_time"] = time.time() - start_time
            
            log_with_context(
                logger, "info",
                "문서 수집 완료",
                total_documents=collection_result["total_documents"],
                processed_documents=collection_result["processed_documents"],
                processing_time=collection_result["processing_time"]
            )
            
        except Exception as e:
            self.collection_stats["failed_collections"] += 1
            self.collection_stats["last_error"] = str(e)
            collection_result["errors"].append(str(e))
            collection_result["processing_time"] = time.time() - start_time
            
            log_with_context(
                logger, "error",
                "문서 수집 실패",
                error=str(e),
                processing_time=collection_result["processing_time"]
            )
        
        return collection_result
    
    async def _extract_documents_from_table(
        self, 
        table_name: str, 
        incremental: bool = False,
        limit: Optional[int] = None
    ) -> List[Document]:
        """테이블에서 문서 추출"""
        try:
            loop = asyncio.get_event_loop()
            
            # 캐시된 스키마 분석 조회 또는 새로 분석
            schema_analysis = self.schema_detector.get_cached_analysis(table_name)
            
            if not schema_analysis:
                # 테이블 스키마 조회
                schema = await loop.run_in_executor(
                    None,
                    self.db_manager.get_table_schema,
                    table_name
                )
                
                # 샘플 데이터로 스키마 분석
                sample_data = await loop.run_in_executor(
                    None,
                    self.db_manager.execute_query,
                    f"SELECT * FROM {table_name} LIMIT 5"
                )
                
                schema_analysis = self.schema_detector.analyze_table_schema(
                    table_name, schema, sample_data
                )
            
            column_mapping = schema_analysis['column_mapping']
            
            # 쿼리 구성
            query = self._build_select_query(
                table_name, column_mapping, incremental, limit
            )
            
            # 데이터 조회
            rows = await loop.run_in_executor(
                None,
                self.db_manager.execute_query,
                query
            )
            
            # Document 객체로 변환 (DocumentExtractor 사용)
            documents = self.document_extractor.extract_and_transform_documents(
                rows, table_name, column_mapping
            )
            
            log_with_context(
                logger, "info",
                "테이블에서 문서 추출 완료",
                table_name=table_name,
                document_count=len(documents)
            )
            
            return documents
            
        except Exception as e:
            log_with_context(
                logger, "error",
                "테이블 문서 추출 실패",
                table_name=table_name,
                error=str(e)
            )
            raise e
    

    
    def _build_select_query(
        self, 
        table_name: str, 
        column_mapping: Dict[str, str],
        incremental: bool = False,
        limit: Optional[int] = None
    ) -> str:
        """SELECT 쿼리 구성"""
        # 선택할 컬럼들
        select_columns = []
        for field, column in column_mapping.items():
            if column:
                select_columns.append(f"{column} as {field}")
        
        # 기본 쿼리
        query = f"SELECT {', '.join(select_columns)} FROM {table_name}"
        
        # WHERE 조건 (증분 업데이트)
        if incremental and column_mapping.get('updated_at'):
            # 마지막 수집 시간 이후의 데이터만 조회
            if self.last_collection_time:
                query += f" WHERE {column_mapping['updated_at']} > '{self.last_collection_time}'"
        
        # ORDER BY
        if column_mapping.get('created_at'):
            query += f" ORDER BY {column_mapping['created_at']} DESC"
        
        # LIMIT
        if limit:
            query += f" LIMIT {limit}"
        
        return query
    

    
    async def close(self):
        """리소스 정리"""
        self.connection_status = "disconnected"
        logger.info("데이터 수집 매니저 종료")