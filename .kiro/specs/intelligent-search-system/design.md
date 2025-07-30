# 설계 문서

## 개요

지능형 검색 시스템은 공공기관 내부 서버에 설치되어 기관 내 RDB와 연결하여 행정·민원·조례 문서를 자동으로 수집, 벡터화, 인덱싱하고 RAG 기반 검색 서비스를 제공하는 마이크로서비스 아키텍처 기반 시스템입니다.

## 아키텍처

### 전체 시스템 아키텍처

```mermaid
graph TB
    subgraph "External Systems"
        DB[(Internal RDB)]
        Admin[System Admin]
        User[End User]
    end
    
    subgraph "API Gateway Layer"
        Gateway[FastAPI Gateway]
    end
    
    subgraph "Core Services"
        DataService[Data Collection Service]
        ProcessService[Document Processing Service]
        VectorService[Vector Service]
        SearchService[Search Service]
        EvalService[Evaluation Service]
    end
    
    subgraph "Storage Layer"
        VectorDB[(Vector Database)]
        MetaDB[(Metadata Database)]
        LogDB[(Log Database)]
    end
    
    subgraph "ML Models"
        EmbedModel[BGE KoBase Model]
        LLMModel[KoAlpaca/LLaMA Model]
    end
    
    subgraph "Scheduler"
        Scheduler[APScheduler]
    end
    
    DB --> DataService
    DataService --> ProcessService
    ProcessService --> VectorService
    VectorService --> VectorDB
    VectorService --> EmbedModel
    
    User --> Gateway
    Gateway --> SearchService
    SearchService --> VectorDB
    SearchService --> LLMModel
    
    Admin --> Gateway
    Gateway --> EvalService
    EvalService --> VectorDB
    EvalService --> LLMModel
    
    Scheduler --> DataService
    Scheduler --> EvalService
    
    ProcessService --> MetaDB
    SearchService --> MetaDB
    EvalService --> LogDB
```

### 마이크로서비스 구조

시스템은 다음과 같은 독립적인 서비스들로 구성됩니다:

1. **API Gateway Service**: 모든 외부 요청의 진입점
2. **Data Collection Service**: 데이터베이스 연결 및 문서 수집
3. **Document Processing Service**: 문서 구조화 및 청킹
4. **Vector Service**: 임베딩 생성 및 벡터 인덱싱
5. **Search Service**: 검색 및 RAG 응답 생성
6. **Evaluation Service**: 성능 평가 및 모니터링
7. **Scheduler Service**: 배치 작업 스케줄링

## 구성요소 및 인터페이스

### 1. API Gateway Service

**기술 스택**: FastAPI, Uvicorn
**포트**: 8000

#### 주요 기능
- 인증 및 권한 관리
- 요청 라우팅 및 로드 밸런싱
- API 문서 자동 생성 (Swagger)
- 요청/응답 로깅

#### API 엔드포인트
```python
# 검색 API
POST /api/v1/search
{
    "query": "민원 처리 절차",
    "document_types": ["행정문서", "민원문서"],
    "top_k": 5
}

# 문서 수집 트리거 API
POST /api/v1/admin/collect
{
    "table_names": ["documents", "regulations"],
    "incremental": true
}

# 평가 실행 API
POST /api/v1/admin/evaluate
{
    "test_dataset": "standard_qa.json"
}

# 시스템 상태 API
GET /api/v1/health
```

### 2. Data Collection Service

**기술 스택**: Python, SQLAlchemy, PyODBC
**포트**: 8001

#### 주요 기능
- JDBC/ODBC 데이터베이스 연결 관리
- 테이블 스키마 자동 감지
- 증분 데이터 수집
- 연결 풀 관리

#### 데이터베이스 연결 설정
```python
class DatabaseConfig:
    driver: str = "ODBC Driver 17 for SQL Server"
    server: str
    database: str
    username: str
    password: str
    connection_pool_size: int = 10
    connection_timeout: int = 30
```

#### 문서 추출 인터페이스
```python
class DocumentExtractor:
    def extract_documents(self, table_config: TableConfig) -> List[Document]:
        """지정된 테이블에서 문서 추출"""
        pass
    
    def detect_schema_changes(self, table_name: str) -> bool:
        """스키마 변경 감지"""
        pass
    
    def get_incremental_updates(self, table_name: str, last_update: datetime) -> List[Document]:
        """증분 업데이트 추출"""
        pass
```

### 3. Document Processing Service

**기술 스택**: Python, spaCy, KoNLPy
**포트**: 8002

#### 주요 기능
- JSON 스키마 자동 매핑
- 한국어 문서 구조 분석
- 의미 단위 청킹
- 문서 타입 분류

#### 문서 처리 파이프라인
```python
class DocumentProcessor:
    def __init__(self):
        self.korean_tokenizer = Okt()  # KoNLPy
        self.sentence_splitter = KoreanSentenceSplitter()
    
    def process_document(self, raw_document: RawDocument) -> ProcessedDocument:
        """문서 전처리 및 구조화"""
        # 1. JSON 스키마 매핑
        structured_doc = self.map_to_schema(raw_document)
        
        # 2. 문서 타입 분류
        doc_type = self.classify_document_type(structured_doc)
        
        # 3. 청킹 수행
        chunks = self.chunk_document(structured_doc, doc_type)
        
        return ProcessedDocument(
            document_id=structured_doc.id,
            document_type=doc_type,
            chunks=chunks,
            metadata=structured_doc.metadata
        )
```

#### 한국어 청킹 전략
```python
class KoreanChunker:
    def __init__(self):
        self.min_chunk_size = 100  # 최소 100자
        self.max_chunk_size = 1000  # 최대 1000자
        self.overlap_size = 50  # 50자 오버랩
    
    def chunk_by_structure(self, document: Document) -> List[Chunk]:
        """구조 기반 청킹 (조항, 항목, 문단 단위)"""
        pass
    
    def chunk_by_semantics(self, document: Document) -> List[Chunk]:
        """의미 기반 청킹 (LLM 보조)"""
        pass
    
    def chunk_by_sentences(self, document: Document) -> List[Chunk]:
        """문장 기반 청킹 (기본 전략)"""
        pass
```

### 4. Vector Service

**기술 스택**: Python, Transformers, Sentence-Transformers
**포트**: 8003

#### 주요 기능
- BGE KoBase 모델 기반 임베딩 생성
- 배치 처리 최적화
- 벡터 데이터베이스 인덱싱
- 모델 버전 관리

#### 임베딩 모델 설정
```python
class EmbeddingConfig:
    primary_model = "BAAI/bge-base-ko-v1.5"  # BGE KoBase
    fallback_model = "klue/roberta-base"     # SentenceTransformer 대안
    vector_dimension = 768
    batch_size = 32
    max_sequence_length = 512
```

#### 벡터화 인터페이스
```python
class VectorService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EmbeddingConfig.primary_model)
        self.vector_db = ChromaClient()
    
    def embed_chunks(self, chunks: List[Chunk]) -> List[Vector]:
        """청크를 벡터로 변환"""
        texts = [chunk.text for chunk in chunks]
        vectors = self.embedding_model.encode(
            texts, 
            batch_size=EmbeddingConfig.batch_size,
            show_progress_bar=True
        )
        return vectors
    
    def index_vectors(self, vectors: List[Vector], metadata: List[Dict]) -> bool:
        """벡터 데이터베이스에 인덱싱"""
        collection = self.vector_db.get_or_create_collection(
            name=f"documents_{metadata[0]['document_type']}"
        )
        collection.add(
            embeddings=vectors,
            metadatas=metadata,
            ids=[f"chunk_{i}" for i in range(len(vectors))]
        )
        return True
```

### 5. Search Service

**기술 스택**: Python, Transformers, ChromaDB
**포트**: 8004

#### 주요 기능
- 벡터 유사도 검색
- LLM 기반 응답 생성
- 프롬프트 템플릿 관리
- 검색 결과 후처리

#### 검색 파이프라인
```python
class SearchService:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EmbeddingConfig.primary_model)
        self.llm_model = self.load_llm_model()  # KoAlpaca or LLaMA
        self.vector_db = ChromaClient()
    
    def search(self, query: str, document_types: List[str], top_k: int = 5) -> SearchResult:
        # 1. 쿼리 벡터화
        query_vector = self.embedding_model.encode([query])
        
        # 2. 벡터 검색
        relevant_chunks = self.vector_search(query_vector, document_types, top_k)
        
        # 3. 신뢰도 필터링 (0.7 이상)
        filtered_chunks = [chunk for chunk in relevant_chunks if chunk.score >= 0.7]
        
        # 4. LLM 응답 생성
        if filtered_chunks:
            response = self.generate_response(query, filtered_chunks)
        else:
            response = "관련 정보를 찾을 수 없습니다."
        
        return SearchResult(
            query=query,
            response=response,
            sources=filtered_chunks,
            confidence_score=max([chunk.score for chunk in filtered_chunks]) if filtered_chunks else 0.0
        )
```

#### 프롬프트 템플릿
```python
PROMPT_TEMPLATE = """
다음은 공공기관의 문서에서 검색된 관련 정보입니다.

질문: {query}

관련 문서:
{context}

위 정보를 바탕으로 질문에 대해 정확하고 공식적인 답변을 제공해주세요.
답변은 다음 형식을 따라주세요:
1. 핵심 답변
2. 관련 근거 (문서명, 조항 등)
3. 추가 참고사항 (있는 경우)

답변:
"""
```

## 데이터 모델

### 문서 데이터 모델
```python
@dataclass
class Document:
    id: str
    title: str
    content: str
    document_type: str  # "행정문서", "민원문서", "조례문서"
    source_table: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    chunk_index: int
    vector: Optional[List[float]]
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    query: str
    response: str
    sources: List[Chunk]
    confidence_score: float
    processing_time: float
```

### 벡터 데이터베이스 스키마 (Chroma)
```python
# 컬렉션별 메타데이터 구조
collection_metadata = {
    "documents_행정문서": {
        "document_id": str,
        "document_title": str,
        "chunk_index": int,
        "source_table": str,
        "created_at": str
    },
    "documents_민원문서": {
        "document_id": str,
        "document_title": str,
        "chunk_index": int,
        "category": str,  # 민원 카테고리
        "created_at": str
    },
    "documents_조례문서": {
        "document_id": str,
        "document_title": str,
        "chunk_index": int,
        "article_number": str,  # 조항 번호
        "created_at": str
    }
}
```

## 오류 처리

### 오류 분류 및 처리 전략

#### 1. 데이터베이스 연결 오류
- **오류 유형**: 연결 실패, 타임아웃, 권한 오류
- **처리 방법**: 지수 백오프 재시도, 연결 풀 재초기화
- **로깅**: ERROR 레벨, 상세 스택 트레이스 포함

#### 2. 문서 처리 오류
- **오류 유형**: 청킹 실패, 인코딩 오류, 스키마 불일치
- **처리 방법**: 대안 청킹 전략 적용, 기본값 사용
- **로깅**: WARN 레벨, 문서 ID 및 오류 원인 기록

#### 3. 벡터화 오류
- **오류 유형**: 모델 로딩 실패, 메모리 부족, 배치 처리 오류
- **처리 방법**: 대안 모델 사용, 배치 크기 축소, 단일 처리 모드
- **로깅**: ERROR 레벨, 성능 지표 포함

#### 4. 검색 서비스 오류
- **오류 유형**: 벡터 DB 연결 실패, LLM 응답 생성 실패
- **처리 방법**: 캐시된 결과 반환, 기본 응답 제공
- **로깅**: ERROR 레벨, 사용자 쿼리 및 컨텍스트 기록

### 오류 모니터링 시스템
```python
class ErrorMonitor:
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.alert_thresholds = {
            "database_connection": 5,  # 5회 연속 실패 시 알림
            "chunking_failure": 10,    # 10회 실패 시 알림
            "vectorization_failure": 3, # 3회 실패 시 알림
        }
    
    def log_error(self, error_type: str, error_details: Dict):
        """오류 로깅 및 임계값 체크"""
        self.error_counts[error_type] += 1
        
        if self.error_counts[error_type] >= self.alert_thresholds.get(error_type, 10):
            self.send_alert(error_type, error_details)
            self.error_counts[error_type] = 0  # 카운터 리셋
```

## 테스팅 전략

### 1. 단위 테스트
- **대상**: 각 서비스의 핵심 로직
- **도구**: pytest, unittest.mock
- **커버리지**: 최소 80% 이상

### 2. 통합 테스트
- **대상**: 서비스 간 인터페이스
- **도구**: pytest-asyncio, testcontainers
- **시나리오**: 전체 파이프라인 테스트

### 3. 성능 테스트
- **대상**: 검색 응답 시간, 처리량
- **도구**: locust, pytest-benchmark
- **목표**: 검색 응답 1초 이내, 동시 사용자 100명 지원

### 4. 품질 테스트
- **대상**: 검색 정확도, LLM 응답 품질
- **도구**: 자체 개발 AutoEval 시스템
- **지표**: Top-5 정확도 80% 이상, BLEU 점수 0.7 이상

### 테스트 데이터셋
```python
# 표준 평가 데이터셋 구조
test_dataset = {
    "questions": [
        {
            "id": "q001",
            "question": "민원 처리 기간은 얼마나 걸리나요?",
            "expected_answer": "일반 민원은 7일 이내, 복잡한 민원은 14일 이내 처리됩니다.",
            "relevant_documents": ["doc_001", "doc_045"],
            "document_type": "민원문서"
        }
    ]
}
```
##
 배포 및 운영

### 컨테이너화 전략
```dockerfile
# 각 서비스별 Docker 이미지
FROM python:3.9-slim

# 의존성 설치
COPY requirements.txt .
RUN pip install -r requirements.txt

# 모델 다운로드 (빌드 시)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-ko-v1.5')"

# 애플리케이션 코드
COPY . /app
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose 구성
```yaml
version: '3.8'
services:
  api-gateway:
    build: ./services/api-gateway
    ports:
      - "8000:8000"
    depends_on:
      - data-service
      - search-service
  
  data-service:
    build: ./services/data-collection
    ports:
      - "8001:8001"
    environment:
      - DB_CONNECTION_STRING=${DB_CONNECTION_STRING}
  
  vector-service:
    build: ./services/vector-service
    ports:
      - "8003:8003"
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  chroma-db:
    image: chromadb/chroma:latest
    ports:
      - "8100:8000"
    volumes:
      - chroma_data:/chroma/chroma
  
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  chroma_data:
```

### 모니터링 및 로깅
```python
# 로깅 설정
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "%(name)s", "message": "%(message)s"}'
        }
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "standard"
        },
        "file": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/var/log/intelligent-search/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": False
        }
    }
}
```

### 성능 최적화

#### 1. 캐싱 전략
```python
# Redis 기반 캐싱
class CacheManager:
    def __init__(self):
        self.redis_client = redis.Redis(host='redis', port=6379, db=0)
        self.cache_ttl = {
            'search_results': 3600,      # 1시간
            'document_embeddings': 86400, # 24시간
            'model_predictions': 1800     # 30분
        }
    
    def get_cached_search(self, query_hash: str) -> Optional[SearchResult]:
        """검색 결과 캐시 조회"""
        cached = self.redis_client.get(f"search:{query_hash}")
        return json.loads(cached) if cached else None
    
    def cache_search_result(self, query_hash: str, result: SearchResult):
        """검색 결과 캐싱"""
        self.redis_client.setex(
            f"search:{query_hash}",
            self.cache_ttl['search_results'],
            json.dumps(result.to_dict())
        )
```

#### 2. 배치 처리 최적화
```python
class BatchProcessor:
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
        self.processing_queue = asyncio.Queue()
    
    async def process_documents_batch(self, documents: List[Document]):
        """문서 배치 처리"""
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            await self.process_batch(batch)
    
    async def process_batch(self, batch: List[Document]):
        """단일 배치 처리"""
        # 병렬 처리로 성능 향상
        tasks = [self.process_single_document(doc) for doc in batch]
        await asyncio.gather(*tasks)
```

#### 3. 모델 최적화
```python
class ModelOptimizer:
    def __init__(self):
        self.model_cache = {}
        self.quantization_enabled = True
    
    def load_optimized_model(self, model_name: str):
        """최적화된 모델 로딩"""
        if model_name not in self.model_cache:
            model = SentenceTransformer(model_name)
            
            # 양자화 적용 (메모리 사용량 감소)
            if self.quantization_enabled:
                model = self.apply_quantization(model)
            
            self.model_cache[model_name] = model
        
        return self.model_cache[model_name]
```

### 보안 고려사항

#### 1. 데이터베이스 보안
- 읽기 전용 계정 사용
- 연결 문자열 암호화
- VPN 또는 전용선을 통한 접근

#### 2. API 보안
```python
# JWT 기반 인증
class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = "HS256"
    
    def verify_token(self, token: str) -> Dict:
        """JWT 토큰 검증"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def create_access_token(self, data: Dict) -> str:
        """액세스 토큰 생성"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=24)
        to_encode.update({"exp": expire})
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
```

#### 3. 데이터 암호화
- 민감한 문서 내용 AES 암호화
- 벡터 데이터베이스 접근 제어
- 로그 파일 암호화

### 확장성 고려사항

#### 1. 수평 확장
- 각 서비스의 독립적인 스케일링
- 로드 밸런서를 통한 트래픽 분산
- 데이터베이스 샤딩 지원

#### 2. 수직 확장
- GPU 리소스 활용 (임베딩 생성)
- 메모리 최적화 (모델 캐싱)
- CPU 병렬 처리 (문서 전처리)

#### 3. 클라우드 네이티브 지원
```yaml
# Kubernetes 배포 예시
apiVersion: apps/v1
kind: Deployment
metadata:
  name: search-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: search-service
  template:
    metadata:
      labels:
        app: search-service
    spec:
      containers:
      - name: search-service
        image: intelligent-search/search-service:latest
        ports:
        - containerPort: 8004
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        env:
        - name: VECTOR_DB_URL
          value: "http://chroma-service:8000"
```

## 결론

이 설계는 공공기관의 내부 문서 검색 요구사항을 충족하면서도 확장 가능하고 유지보수가 용이한 시스템을 제공합니다. 마이크로서비스 아키텍처를 통해 각 구성요소의 독립적인 개발과 배포가 가능하며, 한국어 특화 처리와 공공문서 도메인에 최적화된 기능들을 포함하고 있습니다.

주요 설계 원칙:
- **모듈성**: 각 서비스의 독립적인 개발 및 배포
- **확장성**: 수평/수직 확장 지원
- **안정성**: 오류 처리 및 복구 메커니즘
- **보안성**: 내부 네트워크 기반 보안 설계
- **성능**: 캐싱 및 배치 처리 최적화
- **유지보수성**: 명확한 인터페이스 및 로깅 시스템