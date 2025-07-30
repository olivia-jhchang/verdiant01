# Project Structure

## Directory Organization

```
intelligent-search-system/
├── services/                          # Microservices
│   ├── api-gateway/                   # FastAPI Gateway (port 8000)
│   │   ├── app/
│   │   │   ├── main.py               # FastAPI application entry
│   │   │   ├── routers/              # API route handlers
│   │   │   ├── middleware/           # Auth, logging middleware
│   │   │   └── dependencies.py       # Dependency injection
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── data-collection/              # Database connection service (port 8001)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── database/             # DB connection, schema detection
│   │   │   ├── extractors/           # Document extraction logic
│   │   │   └── models/               # Data models
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── document-processing/          # Korean text processing (port 8002)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── processors/           # Document structuring, chunking
│   │   │   ├── chunkers/             # Korean-specific chunking strategies
│   │   │   └── classifiers/          # Document type classification
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── vector-service/               # Embedding and indexing (port 8003)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── embeddings/           # BGE KoBase, SentenceTransformers
│   │   │   ├── indexing/             # ChromaDB, FAISS operations
│   │   │   └── models/               # Model management
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── search-service/               # RAG search and LLM (port 8004)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── search/               # Vector similarity search
│   │   │   ├── llm/                  # KoAlpaca/LLaMA integration
│   │   │   └── templates/            # Prompt templates
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   ├── evaluation-service/           # AutoEval system (port 8005)
│   │   ├── app/
│   │   │   ├── main.py
│   │   │   ├── evaluators/           # Accuracy, quality metrics
│   │   │   ├── datasets/             # Test question-answer pairs
│   │   │   └── reports/              # Performance reporting
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   │
│   └── scheduler-service/            # APScheduler batch jobs (port 8006)
│       ├── app/
│       │   ├── main.py
│       │   ├── jobs/                 # Scheduled task definitions
│       │   ├── monitoring/           # Job status tracking
│       │   └── recovery/             # Checkpoint and resume logic
│       ├── Dockerfile
│       └── requirements.txt
│
├── shared/                           # Common libraries
│   ├── __init__.py
│   ├── models/                       # Shared data models
│   │   ├── document.py              # Document, Chunk, SearchResult
│   │   ├── config.py                # Configuration classes
│   │   └── exceptions.py            # Custom exceptions
│   ├── utils/                        # Utility functions
│   │   ├── logging.py               # Structured logging setup
│   │   ├── security.py              # JWT, encryption utilities
│   │   └── korean_utils.py          # Korean text processing helpers
│   └── database/                     # Database utilities
│       ├── connection.py            # Connection pooling
│       └── migrations/              # Schema migrations
│
├── tests/                           # Test suites
│   ├── unit/                        # Service-specific unit tests
│   │   ├── test_data_collection.py
│   │   ├── test_document_processing.py
│   │   ├── test_vector_service.py
│   │   └── test_search_service.py
│   ├── integration/                 # Cross-service integration tests
│   │   ├── test_full_pipeline.py
│   │   └── test_api_endpoints.py
│   ├── performance/                 # Load and performance tests
│   │   ├── locustfile.py
│   │   └── benchmark_search.py
│   └── fixtures/                    # Test data and mocks
│       ├── sample_documents.json
│       └── test_database.sql
│
├── config/                          # Configuration files
│   ├── development.env              # Development environment
│   ├── staging.env                  # Staging environment
│   ├── production.env               # Production environment
│   └── logging.yaml                 # Logging configuration
│
├── scripts/                         # Utility scripts
│   ├── setup_database.py           # Database initialization
│   ├── download_models.py          # ML model downloads
│   ├── test_db_connection.py       # Connection testing
│   └── backup_vectors.py           # Vector database backup
│
├── docs/                           # Documentation
│   ├── api/                        # API documentation
│   ├── deployment/                 # Deployment guides
│   └── korean_processing.md        # Korean language processing guide
│
├── docker-compose.yml              # Multi-service orchestration
├── docker-compose.dev.yml          # Development overrides
├── docker-compose.prod.yml         # Production overrides
├── requirements.txt                # Root dependencies
├── pytest.ini                      # Test configuration
└── README.md                       # Project overview
```

## Key Conventions

### Service Structure
- Each service follows the same structure: `app/main.py` as entry point
- Service-specific logic in dedicated subdirectories
- Shared models and utilities in `/shared` directory
- Independent Dockerfiles and requirements per service

### File Naming
- **Python files**: snake_case (e.g., `document_processor.py`)
- **Classes**: PascalCase (e.g., `DocumentProcessor`)
- **Functions/variables**: snake_case (e.g., `process_document()`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_CHUNK_SIZE`)

### Configuration Management
- Environment-specific config files in `/config`
- Database credentials and secrets in environment variables
- Service ports standardized (8000-8006 range)
- Logging configuration centralized in `logging.yaml`

### Korean Language Files
- Korean text processing utilities in `shared/utils/korean_utils.py`
- Korean-specific chunking strategies in `services/document-processing/chunkers/`
- Korean prompt templates in `services/search-service/templates/`
- Test data includes Korean document samples

### Testing Organization
- Unit tests mirror service structure
- Integration tests focus on service interactions
- Performance tests use realistic Korean document datasets
- Fixtures include Korean public document samples

### Documentation Standards
- API documentation auto-generated via FastAPI/Swagger
- Korean language processing documented separately
- Deployment guides for different environments
- Code comments in English, user-facing content in Korean