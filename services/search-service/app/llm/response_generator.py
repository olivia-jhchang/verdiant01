"""
LLM 기반 응답 생성기
"""
import asyncio
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from shared.models.config import LLMConfig
from shared.models.document import Chunk
from shared.models.exceptions import LLMError
from shared.utils.logging import setup_logging, log_with_context

from app.templates.prompt_templates import PromptTemplateManager

logger = setup_logging("response-generator")


class ResponseGenerator:
    """LLM 기반 응답 생성기"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompt_manager = PromptTemplateManager()
        self.is_initialized = False
        
        # 응답 생성 통계
        self.generation_stats = {
            "total_responses_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_generation_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_response_length": 0.0,
            "response_types": {},
            "quality_scores": []
        }
    
    async def initialize(self):
        """응답 생성기 초기화"""
        try:
            logger.info("응답 생성기 초기화 시작")
            
            # 모델 로딩을 별도 스레드에서 실행
            loop = asyncio.get_event_loop()
            
            try:
                # 실제 환경에서는 transformers 사용
                self.model, self.tokenizer = await loop.run_in_executor(
                    None, self._load_llm_model
                )
                logger.info(f"LLM 모델 로딩 완료: {self.config.model_name}")
            except Exception as e:
                logger.warning(f"LLM 모델 로딩 실패, 더미 모델 사용: {e}")
                self.model = self._create_dummy_model()
                self.tokenizer = None
            
            # 프롬프트 템플릿 초기화
            await self.prompt_manager.initialize()
            
            self.is_initialized = True
            
            log_with_context(
                logger, "info",
                "응답 생성기 초기화 완료",
                model_name=self.config.model_name,
                config=self.config.__dict__
            )
            
        except Exception as e:
            logger.error(f"응답 생성기 초기화 실패: {e}")
            raise LLMError(f"초기화 실패: {e}")
    
    def _load_llm_model(self):
        """LLM 모델 로딩"""
        try:
            # 로컬 환경에서는 경량 모델 사용
            if self.config.model_name == "microsoft/DialoGPT-medium":
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    
                    tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                    model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
                    
                    # 패딩 토큰 설정
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    return model, tokenizer
                    
                except ImportError:
                    logger.warning("transformers 미설치, 더미 모델 사용")
                    return self._create_dummy_model(), None
                except Exception as e:
                    logger.warning(f"모델 로딩 실패, 더미 모델 사용: {e}")
                    return self._create_dummy_model(), None
            else:
                return self._create_dummy_model(), None
                
        except Exception as e:
            logger.error(f"LLM 모델 로딩 실패: {e}")
            raise e
    
    def _create_dummy_model(self):
        """더미 LLM 모델 생성 (로컬 테스트용)"""
        class DummyLLMModel:
            def __init__(self):
                logger.info("더미 LLM 모델 생성")
                
                # 미리 정의된 응답 템플릿
                self.response_templates = {
                    "민원": [
                        "민원 처리에 관한 안내를 드립니다. 일반적으로 민원 접수 후 7일 이내에 처리가 완료됩니다.",
                        "민원 신청은 온라인 또는 방문 접수가 가능하며, 필요한 서류를 준비하여 신청하시기 바랍니다.",
                        "민원 처리 현황은 민원 접수증 번호로 확인하실 수 있습니다."
                    ],
                    "조례": [
                        "해당 조례에 따르면, 관련 규정이 다음과 같이 정해져 있습니다.",
                        "조례 제정의 목적은 시민의 편의를 도모하고 행정서비스의 질을 향상시키기 위함입니다.",
                        "조례 시행과 관련하여 자세한 사항은 관련 부서에 문의하시기 바랍니다."
                    ],
                    "행정": [
                        "행정업무 처리에 관한 안내를 드립니다. 관련 절차는 다음과 같습니다.",
                        "업무 처리를 위해서는 해당 부서의 승인이 필요하며, 관련 서류를 준비하시기 바랍니다.",
                        "자세한 사항은 담당 부서에 문의하여 확인하시기 바랍니다."
                    ],
                    "일반": [
                        "문의하신 내용에 대해 안내드립니다.",
                        "관련 정보를 확인하여 답변드리겠습니다.",
                        "추가적인 정보가 필요하시면 관련 부서에 문의하시기 바랍니다."
                    ]
                }
            
            def generate_response(self, prompt: str, context: str) -> str:
                """더미 응답 생성"""
                # 컨텍스트 기반 응답 타입 결정
                response_type = "일반"
                if "민원" in context or "신청" in context:
                    response_type = "민원"
                elif "조례" in context or "제1조" in context:
                    response_type = "조례"
                elif "업무" in context or "행정" in context:
                    response_type = "행정"
                
                # 템플릿에서 응답 선택
                templates = self.response_templates[response_type]
                import random
                base_response = random.choice(templates)
                
                # 컨텍스트 정보 추가
                if context.strip():
                    # 컨텍스트에서 핵심 정보 추출
                    context_lines = context.split('\n')[:2]  # 처음 2줄만 사용
                    context_summary = ' '.join(context_lines).strip()
                    
                    if len(context_summary) > 100:
                        context_summary = context_summary[:100] + "..."
                    
                    response = f"{base_response}\n\n관련 내용: {context_summary}"
                else:
                    response = base_response
                
                return response
        
        return DummyLLMModel()
    
    async def generate_response(
        self, 
        query: str, 
        relevant_chunks: List[Tuple[Chunk, float]],
        response_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """RAG 기반 응답 생성"""
        if not self.is_initialized:
            raise LLMError("응답 생성기가 초기화되지 않았습니다")
        
        start_time = time.time()
        
        try:
            log_with_context(
                logger, "info",
                "응답 생성 시작",
                query=query[:100],
                chunk_count=len(relevant_chunks),
                response_type=response_type
            )
            
            # 1. 컨텍스트 구성
            context = self._build_context(relevant_chunks)
            
            # 2. 프롬프트 생성
            prompt = await self.prompt_manager.build_prompt(
                query, context, response_type
            )
            
            # 3. LLM 응답 생성
            raw_response = await self._generate_llm_response(prompt, context)
            
            # 4. 응답 후처리
            processed_response = self._post_process_response(
                raw_response, query, relevant_chunks
            )
            
            # 5. 품질 평가
            quality_score = self._evaluate_response_quality(
                processed_response, query, context
            )
            
            # 6. 응답 구성
            response_data = {
                "response": processed_response,
                "sources": self._extract_source_info(relevant_chunks),
                "confidence_score": self._calculate_confidence_score(relevant_chunks),
                "quality_score": quality_score,
                "response_type": response_type,
                "processing_time": time.time() - start_time,
                "metadata": {
                    "query_length": len(query),
                    "context_length": len(context),
                    "response_length": len(processed_response),
                    "source_count": len(relevant_chunks)
                }
            }
            
            # 7. 통계 업데이트
            generation_time = time.time() - start_time
            self._update_generation_stats(
                response_type, len(processed_response), generation_time, quality_score, True
            )
            
            log_with_context(
                logger, "info",
                "응답 생성 완료",
                query=query[:100],
                response_length=len(processed_response),
                quality_score=quality_score,
                generation_time=generation_time
            )
            
            return response_data
            
        except Exception as e:
            generation_time = time.time() - start_time
            self._update_generation_stats(
                response_type, 0, generation_time, 0.0, False
            )
            
            logger.error(f"응답 생성 실패: {e}")
            raise LLMError(f"응답 생성 실패: {e}")
    
    def _build_context(self, relevant_chunks: List[Tuple[Chunk, float]]) -> str:
        """검색된 청크들로부터 컨텍스트 구성"""
        if not relevant_chunks:
            return ""
        
        context_parts = []
        
        for i, (chunk, score) in enumerate(relevant_chunks):
            # 소스 정보 추가
            source_info = ""
            if chunk.metadata:
                doc_type = chunk.metadata.get("document_type", "")
                doc_title = chunk.metadata.get("document_title", "")
                if doc_title:
                    source_info = f"[{doc_type}: {doc_title}]"
                elif doc_type:
                    source_info = f"[{doc_type}]"
            
            # 청크 내용 추가
            context_part = f"{source_info}\n{chunk.text.strip()}"
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    async def _generate_llm_response(self, prompt: str, context: str) -> str:
        """LLM을 사용한 응답 생성"""
        try:
            if hasattr(self.model, 'generate_response'):
                # 더미 모델 사용
                return self.model.generate_response(prompt, context)
            
            elif self.tokenizer is not None:
                # 실제 transformers 모델 사용
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, self._generate_with_transformers, prompt
                )
                return response
            
            else:
                # 기본 응답
                return "죄송합니다. 현재 응답을 생성할 수 없습니다. 관련 부서에 문의해 주시기 바랍니다."
                
        except Exception as e:
            logger.error(f"LLM 응답 생성 실패: {e}")
            return "죄송합니다. 응답 생성 중 오류가 발생했습니다."
    
    def _generate_with_transformers(self, prompt: str) -> str:
        """Transformers를 사용한 응답 생성"""
        try:
            # 입력 토큰화
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # 응답 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + self.config.max_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 디코딩
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 프롬프트 부분 제거
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Transformers 응답 생성 실패: {e}")
            return "응답 생성 중 오류가 발생했습니다."
    
    def _post_process_response(
        self, 
        raw_response: str, 
        query: str, 
        relevant_chunks: List[Tuple[Chunk, float]]
    ) -> str:
        """응답 후처리"""
        try:
            # 기본 정리
            response = raw_response.strip()
            
            # 불필요한 반복 제거
            response = re.sub(r'(.{10,}?)\1+', r'\1', response)
            
            # 문장 완결성 확인
            if response and not response.endswith(('.', '!', '?', '다', '요', '습니다')):
                response += "."
            
            # 길이 제한
            if len(response) > 1000:
                sentences = response.split('.')
                truncated = []
                current_length = 0
                
                for sentence in sentences:
                    if current_length + len(sentence) > 900:
                        break
                    truncated.append(sentence)
                    current_length += len(sentence)
                
                response = '.'.join(truncated)
                if not response.endswith('.'):
                    response += '.'
            
            # 빈 응답 처리
            if not response or len(response) < 10:
                response = "죄송합니다. 관련 정보를 찾을 수 없습니다. 자세한 사항은 관련 부서에 문의해 주시기 바랍니다."
            
            return response
            
        except Exception as e:
            logger.warning(f"응답 후처리 실패: {e}")
            return raw_response
    
    def _evaluate_response_quality(
        self, 
        response: str, 
        query: str, 
        context: str
    ) -> float:
        """응답 품질 평가"""
        try:
            quality_score = 0.0
            
            # 1. 길이 적절성 (0.2)
            length_score = min(len(response) / 200, 1.0)  # 200자 기준
            quality_score += length_score * 0.2
            
            # 2. 완결성 (0.3)
            completeness_score = 1.0 if response.endswith(('.', '!', '?', '다', '요', '습니다')) else 0.5
            quality_score += completeness_score * 0.3
            
            # 3. 관련성 (0.3)
            query_words = set(query.split())
            response_words = set(response.split())
            if query_words:
                relevance_score = len(query_words & response_words) / len(query_words)
            else:
                relevance_score = 0.5
            quality_score += relevance_score * 0.3
            
            # 4. 정보성 (0.2)
            info_keywords = ['절차', '방법', '기간', '서류', '부서', '문의']
            info_score = min(
                sum(1 for keyword in info_keywords if keyword in response) / len(info_keywords),
                1.0
            )
            quality_score += info_score * 0.2
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            logger.warning(f"품질 평가 실패: {e}")
            return 0.5
    
    def _extract_source_info(self, relevant_chunks: List[Tuple[Chunk, float]]) -> List[Dict[str, Any]]:
        """소스 정보 추출"""
        sources = []
        
        for chunk, score in relevant_chunks:
            source_info = {
                "chunk_id": chunk.id,
                "document_id": chunk.document_id,
                "similarity_score": score,
                "text_preview": chunk.text[:100] + "..." if len(chunk.text) > 100 else chunk.text
            }
            
            if chunk.metadata:
                source_info.update({
                    "document_type": chunk.metadata.get("document_type", ""),
                    "document_title": chunk.metadata.get("document_title", ""),
                    "source_table": chunk.metadata.get("source_table", "")
                })
            
            sources.append(source_info)
        
        return sources
    
    def _calculate_confidence_score(self, relevant_chunks: List[Tuple[Chunk, float]]) -> float:
        """신뢰도 점수 계산"""
        if not relevant_chunks:
            return 0.0
        
        # 최고 유사도 점수와 평균 유사도 점수의 가중 평균
        scores = [score for _, score in relevant_chunks]
        max_score = max(scores)
        avg_score = sum(scores) / len(scores)
        
        confidence = (max_score * 0.7) + (avg_score * 0.3)
        return min(confidence, 1.0)
    
    def _update_generation_stats(
        self, 
        response_type: str, 
        response_length: int, 
        generation_time: float,
        quality_score: float,
        success: bool
    ):
        """응답 생성 통계 업데이트"""
        self.generation_stats["total_responses_generated"] += 1
        self.generation_stats["total_generation_time"] += generation_time
        
        if success:
            self.generation_stats["successful_generations"] += 1
            
            # 응답 타입별 통계
            if response_type not in self.generation_stats["response_types"]:
                self.generation_stats["response_types"][response_type] = 0
            self.generation_stats["response_types"][response_type] += 1
            
            # 평균 응답 길이 계산
            total_successful = self.generation_stats["successful_generations"]
            prev_avg_length = self.generation_stats["avg_response_length"]
            
            self.generation_stats["avg_response_length"] = (
                (prev_avg_length * (total_successful - 1) + response_length) / total_successful
            )
            
            # 품질 점수 저장
            self.generation_stats["quality_scores"].append(quality_score)
            if len(self.generation_stats["quality_scores"]) > 100:
                self.generation_stats["quality_scores"] = self.generation_stats["quality_scores"][-100:]
            
        else:
            self.generation_stats["failed_generations"] += 1
        
        # 평균 생성 시간 계산
        if self.generation_stats["successful_generations"] > 0:
            self.generation_stats["avg_generation_time"] = (
                self.generation_stats["total_generation_time"] / 
                self.generation_stats["successful_generations"]
            )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """응답 생성 통계 조회"""
        stats = self.generation_stats.copy()
        
        # 평균 품질 점수 계산
        if stats["quality_scores"]:
            stats["avg_quality_score"] = sum(stats["quality_scores"]) / len(stats["quality_scores"])
        else:
            stats["avg_quality_score"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """통계 초기화"""
        self.generation_stats = {
            "total_responses_generated": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_generation_time": 0.0,
            "avg_generation_time": 0.0,
            "avg_response_length": 0.0,
            "response_types": {},
            "quality_scores": []
        }
    
    async def cleanup(self):
        """리소스 정리"""
        try:
            # 모델 메모리 해제
            self.model = None
            self.tokenizer = None
            self.is_initialized = False
            
            logger.info("응답 생성기 정리 완료")
            
        except Exception as e:
            logger.error(f"응답 생성기 정리 실패: {e}")