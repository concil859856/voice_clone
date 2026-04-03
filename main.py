#!/usr/bin/env python3
"""
Voice-clone HTTP service: reference audio + reference transcript + target text → WAV.

Run:  python main.py
Docs: http://<host>:<port>/docs
"""
from __future__ import annotations

import asyncio
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration (edit here)
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8088,
    "tts_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "tts_language": "English",
    "tts_max_new_tokens": 2048,
    # Hugging Face token for gated models; None → use HF_TOKEN / HUGGINGFACE_HUB_TOKEN env
    "hf_token": None,
    "preload_tts": False,
}
# ---------------------------------------------------------------------------

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.responses import Response
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig


def _hf_token() -> Optional[str]:
    t = CONFIG.get("hf_token")
    if t:
        return str(t)
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def _set_config_dtype_tree(config: PretrainedConfig, dtype: torch.dtype) -> None:
    config.dtype = dtype
    for key in getattr(config, "sub_configs", None) or {}:
        child = getattr(config, key, None)
        if isinstance(child, PretrainedConfig):
            _set_config_dtype_tree(child, dtype)


def _pick_attn_implementation() -> str:
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _load_audio_from_bytes(data: bytes, filename: str = "clip") -> Tuple[np.ndarray, int]:
    buf = io.BytesIO(data)
    try:
        wav, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        wav, sr = librosa.load(buf, sr=None, mono=True)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32, copy=False), int(sr)


class _TTSCloneEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._model: Optional[Qwen3TTSModel] = None

    def _load(self) -> Qwen3TTSModel:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            model_id = CONFIG["tts_model_id"]
            token = _hf_token()
            if torch.cuda.is_available():
                device_map, dtype = "cuda:0", torch.bfloat16
            else:
                device_map, dtype = "cpu", torch.float32
            AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
            model_config = AutoConfig.from_pretrained(model_id, token=token)
            _set_config_dtype_tree(model_config, dtype)
            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                config=model_config,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=_pick_attn_implementation(),
                token=token,
            )
            return self._model

    def ensure_loaded(self) -> None:
        self._load()

    def clone_to_wav_bytes(
        self,
        ref_audio_bytes: bytes,
        ref_filename: str,
        ref_text: str,
        target_text: str,
    ) -> bytes:
        model = self._load()
        wav_np, sr = _load_audio_from_bytes(ref_audio_bytes, ref_filename)
        out_wavs, sample_rate = model.generate_voice_clone(
            text=target_text,
            language=CONFIG["tts_language"],
            ref_audio=(wav_np, sr),
            ref_text=ref_text,
            max_new_tokens=int(CONFIG["tts_max_new_tokens"]),
        )
        out_buf = io.BytesIO()
        sf.write(out_buf, out_wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue()


_engine: Optional[_TTSCloneEngine] = None


def _get_engine() -> _TTSCloneEngine:
    global _engine
    if _engine is None:
        _engine = _TTSCloneEngine()
    return _engine


@asynccontextmanager
async def _lifespan(app: FastAPI):
    if CONFIG.get("preload_tts"):
        await asyncio.to_thread(_get_engine().ensure_loaded)
    yield


app = FastAPI(title="Voice clone", version="1.0.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/voice-clone", response_class=Response)
async def voice_clone(
    reference_audio: UploadFile = File(..., description="Reference speaker clip"),
    ref_text: str = Form(..., description="Transcript of the reference clip"),
    target_text: str = Form(..., description="Text to speak in that voice"),
):
    """
    Multipart form: `reference_audio` (file), `ref_text`, `target_text`.
    Response body: WAV (audio/wav).
    """
    ref_key = ref_text.strip()
    tgt_key = target_text.strip()
    if not ref_key or not tgt_key:
        raise HTTPException(status_code=400, detail="ref_text and target_text must be non-empty")

    raw = await reference_audio.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty reference_audio")
    name = reference_audio.filename or "reference.wav"

    try:
        wav_bytes = await asyncio.to_thread(
            _get_engine().clone_to_wav_bytes,
            raw,
            name,
            ref_key,
            tgt_key,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'attachment; filename="voice_clone.wav"'},
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=str(CONFIG["host"]),
        port=int(CONFIG["port"]),
        log_level="info",
    )
