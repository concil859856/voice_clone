from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from .config import get_settings
from .schemas import TranscribeResponse, VoiceCloneResponse
from .services.asr_client import transcribe_audio_bytes
from .services.tts_engine import TTSCloneEngine

_engine: Optional[TTSCloneEngine] = None


def get_engine() -> TTSCloneEngine:
    global _engine
    if _engine is None:
        _engine = TTSCloneEngine(get_settings())
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.preload_tts:
        await asyncio.to_thread(get_engine().ensure_loaded)
    yield


app = FastAPI(title="Voice clone API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/v1/transcribe", response_model=TranscribeResponse)
async def transcribe(reference_audio: UploadFile = File(..., description="Audio to transcribe (ASR)")):
    """
    Transcribe reference audio using the external Qwen3-ASR vLLM server
    (`VLLM_ASR_BASE_URL`, e.g. `vllm serve Qwen/Qwen3-ASR-1.7B`).
    """
    settings = get_settings()
    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty audio upload")
    name = reference_audio.filename or "reference.wav"
    try:
        text = await transcribe_audio_bytes(settings, raw, filename=name)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ASR service error: {e!s}") from e
    return TranscribeResponse(text=text)


@app.post("/api/v1/voice-clone", response_model=VoiceCloneResponse)
async def voice_clone(
    reference_audio: UploadFile = File(..., description="User's voice reference clip"),
    text: str = Form(..., description="Desired utterance (target transcript)"),
    reference_transcript: Optional[str] = Form(
        None,
        description="Verbatim transcript of reference_audio; if omitted, ASR is used",
    ),
    language: Optional[str] = Form(None, description="TTS language (default from env)"),
):
    """
    Voice cloning: ref audio + what to say. For best quality, the model needs the
    exact words spoken in the reference clip (`reference_transcript`). If you omit it,
    this endpoint calls the ASR server to obtain it.
    """
    settings = get_settings()
    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty reference audio")

    ref_name = reference_audio.filename or "reference.wav"
    lang = language or settings.tts_language_default

    if reference_transcript and reference_transcript.strip():
        ref_text = reference_transcript.strip()
        source = "provided"
    else:
        try:
            ref_text = await transcribe_audio_bytes(settings, raw, filename=ref_name)
        except Exception as e:
            raise HTTPException(
                status_code=502,
                detail=f"ASR failed (provide reference_transcript or fix VLLM ASR): {e!s}",
            ) from e
        source = "asr"

    engine = get_engine()
    try:
        wav_bytes, sr = await asyncio.to_thread(
            engine.clone_to_wav_bytes,
            raw,
            ref_name,
            ref_text,
            text,
            lang,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice clone failed: {e!s}") from e

    b64 = base64.standard_b64encode(wav_bytes).decode("ascii")
    return VoiceCloneResponse(
        reference_transcript=ref_text,
        transcript_source=source,
        text_spoken=text,
        sample_rate=sr,
        audio_base64=b64,
    )