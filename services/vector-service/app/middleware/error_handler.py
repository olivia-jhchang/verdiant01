"""
에러 처리 미들웨어
"""
import traceback
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from shared.models.exceptions import (
    VectorizationError,
    VectorDBError,
    IntelligentSearchException
)
from shared.utils.logging import setup_logging, log_with_context

logger = setup_logging("error-handler")


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """에러 처리 미들웨어"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
            
        except HTTPException as e:
            # FastAPI HTTPException은 그대로 전달
            raise e
            
        except VectorizationError as e:
            log_with_context(
                logger, "error",
                "벡터화 오류",
                error_code=e.error_code,
                message=e.message,
                path=str(request.url)
            )
            
            return JSONResponse(
                status_code=422,
                content={
                    "error": "벡터화 오류",
                    "message": e.message,
                    "error_code": e.error_code,
                    "path": str(request.url)
                }
            )
            
        except VectorDBError as e:
            log_with_context(
                logger, "error",
                "벡터 데이터베이스 오류",
                error_code=e.error_code,
                message=e.message,
                path=str(request.url)
            )
            
            return JSONResponse(
                status_code=503,
                content={
                    "error": "벡터 데이터베이스 오류",
                    "message": e.message,
                    "error_code": e.error_code,
                    "path": str(request.url)
                }
            )
            
        except IntelligentSearchException as e:
            log_with_context(
                logger, "error",
                "시스템 오류",
                error_code=e.error_code,
                message=e.message,
                path=str(request.url)
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "시스템 오류",
                    "message": e.message,
                    "error_code": e.error_code,
                    "path": str(request.url)
                }
            )
            
        except Exception as e:
            # 예상하지 못한 오류
            error_id = f"ERR_{hash(str(e)) % 10000:04d}"
            
            log_with_context(
                logger, "error",
                "예상하지 못한 오류",
                error_id=error_id,
                error_type=type(e).__name__,
                message=str(e),
                traceback=traceback.format_exc(),
                path=str(request.url)
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "내부 서버 오류",
                    "message": "예상하지 못한 오류가 발생했습니다",
                    "error_id": error_id,
                    "path": str(request.url)
                }
            )