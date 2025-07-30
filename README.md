# 지능형 검색 시스템 (Intelligent Search System)

공공기관 내부 서버에 설치되어 기관 내 RDB와 연결하여 행정·민원·조례 문서를 자동으로 수집, 벡터화, 인덱싱하고 RAG 기반 검색 서비스를 제공하는 시스템입니다.

## 주요 기능

- **내부 전용 운영**: 외부 네트워크 의존성 없이 내부 데이터베이스만 활용
- **한국어 최적화**: 한국어 공공문서에 특화된 처리 및 의미 단위 청킹
- **RAG 기반 검색**: 벡터 유사도 검색과 LLM 응답 생성을 결합한 정확한 답변 제공
- **문서 타입 분류**: 행정문서, 민원문서, 조례문서 자동 분류 및 처리
- **자동 평가**: AutoEval 시스템을 통한 지속적인 성능 모니터링

## 시스템 아키텍처

마이크로서비스 아키텍처 기반으로 다음 서비스들로 구성됩니다:

- **API Gateway** (포트 8000): 통합 API 진입점
- **Data Collection Service** (포트 8001): 데이터베이스 연결 및 문서 수집
- **Document Processing Service** (포트 8002): 한국어 문서 처리 및 청킹
- **Vector Service** (포트 8003): 임베딩 생성 및 벡터 인덱싱
- **Search Service** (포트 8004): RAG 검색 및 LLM 응답 생성
- **Evaluation Service** (포트 8005): 성능 평가 및 모니터링
- **Scheduler Service** (포트 8006): 배치 작업 스케줄링

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd intelligent-search-system

# 환경 변수 설정
cp config/development.env .env
```

### 2. Docker를 사용한 실행

```bash
# 개발 환경에서 실행
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# 로그 확인
docker-compose logs -f
```

### 3. 로컬 개발 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 각 서비스 개별 실행 (개발용)
cd services/api-gateway && uvicorn app.main:app --port 8000 --reload
cd services/data-collection && uvicorn app.main:app --port 8001 --reload
# ... 다른 서비스들도 동일하게
```

## API 사용법

### 문서 검색

```bash
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "민원 처리 기간은 얼마나 걸리나요?",
    "document_types": ["민원문서"],
    "top_k": 5
  }'
```

### 문서 수집 트리거

```bash
curl -X POST "http://localhost:8000/api/v1/admin/collect" \
  -H "Content-Type: application/json" \
  -d '{
    "table_names": ["documents"],
    "incremental": true
  }'
```

### 시스템 상태 확인

```bash
curl http://localhost:8000/api/v1/health
```

## 개발 가이드

### 프로젝트 구조

```
intelligent-search-system/
├── services/                   # 마이크로서비스들
│   ├── api-gateway/           # API 게이트웨이
│   ├── data-collection/       # 데이터 수집 서비스
│   ├── document-processing/   # 문서 처리 서비스
│   ├── vector-service/        # 벡터 서비스
│   ├── search-service/        # 검색 서비스
│   ├── evaluation-service/    # 평가 서비스
│   └── scheduler-service/     # 스케줄러 서비스
├── shared/                    # 공통 라이브러리
│   ├── models/               # 데이터 모델
│   ├── utils/                # 유틸리티 함수
│   └── database/             # 데이터베이스 유틸리티
├── tests/                    # 테스트 코드
├── config/                   # 설정 파일
└── docs/                     # 문서
```

### 테스트 실행

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함 테스트
pytest --cov=shared --cov=services --cov-report=html

# 특정 서비스 테스트
pytest tests/unit/test_data_collection.py -v
```

### 코드 품질 검사

```bash
# 코드 포맷팅
black .

# 린팅
flake8 .

# 타입 체크
mypy .
```

## 배포

### 프로덕션 환경

```bash
# 프로덕션 설정으로 실행
docker-compose -f docker-compose.yml up -d

# 스케일링
docker-compose up -d --scale search-service=3
```

### 모니터링

- **헬스체크**: `GET /api/v1/health`
- **메트릭**: Prometheus 메트릭 수집 지원
- **로그**: 구조화된 JSON 로그 출력

## 기술 스택

- **백엔드**: FastAPI, Python 3.9+
- **데이터베이스**: SQLite (개발), PostgreSQL (프로덕션)
- **벡터 DB**: ChromaDB, FAISS
- **캐싱**: Redis
- **ML 모델**: BGE KoBase, SentenceTransformers, KoAlpaca
- **컨테이너**: Docker, Docker Compose
- **한국어 처리**: KoNLPy, spaCy

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 기여

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해 주세요.