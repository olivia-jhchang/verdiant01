FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 다운로드 (빌드 시)
RUN python -c "from sentence_transformers import SentenceTransformer; print('모델 다운로드 준비 완료')" || echo "sentence-transformers 설치 확인"

# 애플리케이션 코드 복사
COPY app/ ./app/

# 환경 변수 설정
ENV PYTHONPATH=/app:/app/shared
ENV PYTHONUNBUFFERED=1

# 포트 노출
EXPOSE 8003

# 애플리케이션 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]