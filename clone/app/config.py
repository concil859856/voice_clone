from dataclasses import dataclass
import os
from functools import lru_cache
from typing import Optional


@dataclass(frozen=True)
class Settings:
    vllm_asr_base_url: str
    vllm_asr_model: str
    vllm_api_key: str
    tts_model_id: str
    tts_language_default: str
    tts_max_new_tokens: int
    hf_token: Optional[str]
    preload_tts: bool
    host: str
    port: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        # OpenAI-compatible base URL for ASR (override with env VLLM_ASR_BASE_URL).
        vllm_asr_base_url=os.getenv(
            "VLLM_ASR_BASE_URL", "http://144.202.61.73:8099/v1"
        ).rstrip("/"),
        vllm_asr_model=os.getenv("VLLM_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        tts_model_id=os.getenv("TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        tts_language_default=os.getenv("TTS_LANGUAGE_DEFAULT", "English"),
        tts_max_new_tokens=int(os.getenv("TTS_MAX_NEW_TOKENS", "2048")),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        preload_tts=os.getenv("PRELOAD_TTS", "").lower() in ("1", "true", "yes"),
        host=os.getenv("CLONE_API_HOST", "0.0.0.0"),
        port=int(os.getenv("CLONE_API_PORT", "8088")),
    )
