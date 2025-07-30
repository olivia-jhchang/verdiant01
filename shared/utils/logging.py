"""
구조화된 로깅 설정
"""
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any


class JSONFormatter(logging.Formatter):
    """JSON 형식 로그 포매터"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # 추가 필드가 있으면 포함
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(service_name: str, log_level: str = "INFO") -> logging.Logger:
    """서비스별 로깅 설정"""
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 콘솔 핸들러 추가
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (로컬 개발용)
    try:
        file_handler = logging.FileHandler(f"logs/{service_name}.log")
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)
    except FileNotFoundError:
        # logs 디렉토리가 없으면 생성하지 않고 콘솔만 사용
        pass
    
    return logger


def log_with_context(logger: logging.Logger, level: str, message: str, **context):
    """컨텍스트 정보와 함께 로깅"""
    extra = {"extra_fields": context}
    getattr(logger, level.lower())(message, extra=extra)