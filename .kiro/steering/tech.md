# Technology Stack

## Architecture

**Microservices Architecture** - Independent services for scalability and maintainability:
- API Gateway Service (FastAPI, port 8000)
- Data Collection Service (Python, SQLAlchemy, port 8001) 
- Document Processing Service (Python, spaCy, KoNLPy, port 8002)
- Vector Service (Python, Transformers, port 8003)
- Search Service (Python, ChromaDB, port 8004)
- Evaluation Service (AutoEval framework)
- Scheduler Service (APScheduler)

## Core Technologies

### Backend Framework
- **FastAPI** - Primary web framework for all services
- **Uvicorn** - ASGI server
- **SQLAlchemy** - Database ORM
- **PyODBC** - Database connectivity

### Korean Language Processing
- **KoNLPy** - Korean natural language processing
- **spaCy** - Text processing pipeline
- **Okt (Open Korean Text)** - Korean tokenization

### Machine Learning & AI
- **BGE KoBase Model** (`BAAI/bge-base-ko-v1.5`) - Primary embedding model (768 dimensions)
- **SentenceTransformers** - Alternative embedding (`klue/roberta-base`)
- **KoAlpaca/LLaMA 2** - LLM for response generation
- **Transformers** - Model loading and inference

### Vector Database & Storage
- **ChromaDB** - Primary vector database with metadata filtering
- **FAISS** - Alternative vector indexing (IVF_FLAT, HNSW)
- **Redis** - Caching layer
- **PostgreSQL/SQL Server** - Metadata and logging storage

### Containerization & Deployment
- **Docker** - Service containerization
- **Docker Compose** - Multi-service orchestration
- **Kubernetes** - Production deployment (optional)

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download Korean models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-ko-v1.5')"

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f [service-name]
```

### Testing
```bash
# Run unit tests
pytest tests/ -v --cov=src --cov-report=html

# Run integration tests
pytest tests/integration/ -v

# Performance testing
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

### Database Operations
```bash
# Test database connection
python scripts/test_db_connection.py

# Run document collection
curl -X POST "http://localhost:8000/api/v1/admin/collect" -H "Content-Type: application/json" -d '{"incremental": true}'

# Trigger evaluation
curl -X POST "http://localhost:8000/api/v1/admin/evaluate"
```

### Monitoring
```bash
# Check service health
curl http://localhost:8000/api/v1/health

# View system metrics
docker stats

# Check vector database status
curl http://localhost:8100/api/v1/heartbeat
```

## Configuration Standards

### Environment Variables
- Database connections encrypted and stored in environment files
- Model paths and cache directories configurable
- Batch sizes and performance parameters tunable
- Logging levels per service

### Batch Processing
- Default batch size: 32 for vectorization
- Chunk size limits: 100-1000 characters
- Overlap size: 50 characters
- Processing checkpoints for resumability

### Performance Targets
- Search response time: < 1 second
- Concurrent users: 100+
- Top-5 accuracy: > 80%
- Vector similarity threshold: > 0.7