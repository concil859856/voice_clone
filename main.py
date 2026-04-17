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
from pathlib import Path
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
    # Hard timeout per HTTP request for generation work.
    "request_timeout_seconds": 50,
    # Fixed-voice profiles used by /clone_const and other clone_* endpoints:
    "fixed_voice_dir": "/workspace/bt_voices",
    "fixed_voice_scripts": {
        "const": "private so that you can rent a machine on the cloud and you can do all your work on it, but nobody can ever peer inside it. And that is, that is like cryptographic",
        "assistant": "Vocence is a Bittensor subnet that focusing on voice intelligence on decentalized netowrk. it is really cool to be honest",
        "mark": "Experts that cough up the next thing because of the sort of groupthink mindset that just infects the universities and the",
        "nova": "Something that is truly innovative, and that was, um, one of the winning code submissions for Nova Blueprint used this optimization strategy",
        "joker": "Oh wow, Haha. Is that so funny? then I can give you more jokes really, that is not a big deal to me",
    },
    # Optional per-voice path override (empty => auto-detect from fixed_voice_dir).
    "fixed_voice_paths": {},
    # Simple token auth used by fixed clone endpoints:
    "const_auth_token": "vexor_bot_token",
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
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
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


def _encode_audio_bytes(wav: np.ndarray, sample_rate: int, output_format: str) -> Tuple[bytes, str, str]:
    fmt = output_format.strip().lower()
    out_buf = io.BytesIO()
    if fmt == "wav":
        sf.write(out_buf, wav, sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue(), "audio/wav", "wav"
    if fmt == "ogg":
        # Encodes to OGG Vorbis directly from model waveform (no intermediate WAV file).
        sf.write(out_buf, wav, sample_rate, format="OGG", subtype="VORBIS")
        return out_buf.getvalue(), "audio/ogg", "ogg"
    raise ValueError("output_format must be either 'wav' or 'ogg'")


def _resolve_fixed_voice_audio_path(voice_name: str) -> Path:
    fixed_paths = CONFIG.get("fixed_voice_paths", {})
    if isinstance(fixed_paths, dict):
        p = fixed_paths.get(voice_name)
        if p:
            ref_path = Path(str(p))
            if ref_path.is_file():
                return ref_path
            raise FileNotFoundError(f"Configured fixed voice path not found for '{voice_name}': {ref_path}")

    base_dir = Path(str(CONFIG.get("fixed_voice_dir", "/workspace/bt_voices")))
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Fixed voice directory not found: {base_dir}")

    audio_exts = {".wav", ".mp3", ".ogg", ".m4a", ".flac", ".webm"}
    voice_key = voice_name.lower()
    matches = [
        p
        for p in base_dir.iterdir()
        if p.is_file() and p.suffix.lower() in audio_exts and voice_key in p.stem.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No audio file found for '{voice_name}' in {base_dir}. "
            f"Expected filename containing '{voice_name}' (e.g. {voice_name}.wav)."
        )
    matches.sort(key=lambda p: (len(p.name), p.name.lower()))
    return matches[0]


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

    def clone_audio(
        self,
        ref_audio_bytes: bytes,
        ref_filename: str,
        ref_text: str,
        target_text: str,
    ) -> Tuple[np.ndarray, int]:
        model = self._load()
        wav_np, sr = _load_audio_from_bytes(ref_audio_bytes, ref_filename)
        out_wavs, sample_rate = model.generate_voice_clone(
            text=target_text,
            language=CONFIG["tts_language"],
            ref_audio=(wav_np, sr),
            ref_text=ref_text,
            max_new_tokens=int(CONFIG["tts_max_new_tokens"]),
        )
        return out_wavs[0], int(sample_rate)


_engine: Optional[_TTSCloneEngine] = None


def _get_engine() -> _TTSCloneEngine:
    global _engine
    if _engine is None:
        _engine = _TTSCloneEngine()
    return _engine


async def _clone_with_timeout(
    ref_audio_bytes: bytes,
    ref_filename: str,
    ref_text: str,
    target_text: str,
) -> Tuple[np.ndarray, int]:
    timeout_s = float(CONFIG.get("request_timeout_seconds", 50))
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(
                _get_engine().clone_audio,
                ref_audio_bytes,
                ref_filename,
                ref_text,
                target_text,
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Generation exceeded timeout of {timeout_s:.0f}s; request aborted.",
        ) from e


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
        wav_np, sample_rate = await _clone_with_timeout(
            raw,
            name,
            ref_key,
            tgt_key,
        )
        audio_bytes, media_type, ext = _encode_audio_bytes(wav_np, sample_rate, "wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="voice_clone.{ext}"'},
    )


@app.post("/clone_const", response_class=Response)
async def clone_const(
    text: str = Form(..., description="Text to speak using const_reborn voice"),
    output_format: str = Form("wav", description="Output audio format: wav or ogg"),
    x_token: Optional[str] = Header(None, alias="X-Token"),
):
    return await _clone_fixed_voice("const", text, output_format, x_token)


async def _clone_fixed_voice(
    voice_name: str,
    text: str,
    output_format: str,
    x_token: Optional[str],
) -> Response:
    target_text = text.strip()
    if not target_text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    expected_token = str(CONFIG.get("const_auth_token", "")).strip()
    if not expected_token:
        raise HTTPException(status_code=500, detail="CONFIG['const_auth_token'] is empty")
    if x_token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")

    scripts = CONFIG.get("fixed_voice_scripts", {})
    ref_text = str(scripts.get(voice_name, "")).strip() if isinstance(scripts, dict) else ""
    if not ref_text:
        raise HTTPException(
            status_code=500,
            detail=f"Missing fixed script in CONFIG['fixed_voice_scripts'] for '{voice_name}'",
        )

    try:
        ref_path = _resolve_fixed_voice_audio_path(voice_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    raw = ref_path.read_bytes()
    if not raw:
        raise HTTPException(status_code=500, detail=f"Fixed reference audio is empty: {ref_path}")

    try:
        wav_np, sample_rate = await _clone_with_timeout(
            raw,
            ref_path.name,
            ref_text,
            target_text,
        )
        audio_bytes, media_type, ext = _encode_audio_bytes(wav_np, sample_rate, output_format)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="clone_{voice_name}.{ext}"'},
    )


@app.post("/clone_assistant", response_class=Response)
async def clone_assistant(
    text: str = Form(..., description="Text to speak using assistant voice"),
    output_format: str = Form("wav", description="Output audio format: wav or ogg"),
    x_token: Optional[str] = Header(None, alias="X-Token"),
):
    return await _clone_fixed_voice("assistant", text, output_format, x_token)


@app.post("/clone_mark", response_class=Response)
async def clone_mark(
    text: str = Form(..., description="Text to speak using mark voice"),
    output_format: str = Form("wav", description="Output audio format: wav or ogg"),
    x_token: Optional[str] = Header(None, alias="X-Token"),
):
    return await _clone_fixed_voice("mark", text, output_format, x_token)


@app.post("/clone_nova", response_class=Response)
async def clone_nova(
    text: str = Form(..., description="Text to speak using nova voice"),
    output_format: str = Form("wav", description="Output audio format: wav or ogg"),
    x_token: Optional[str] = Header(None, alias="X-Token"),
):
    return await _clone_fixed_voice("nova", text, output_format, x_token)


@app.post("/clone_joker", response_class=Response)
async def clone_joker(
    text: str = Form(..., description="Text to speak using joker voice"),
    output_format: str = Form("wav", description="Output audio format: wav or ogg"),
    x_token: Optional[str] = Header(None, alias="X-Token"),
):
    return await _clone_fixed_voice("joker", text, output_format, x_token)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=str(CONFIG["host"]),
        port=int(CONFIG["port"]),
        log_level="info",
    )
