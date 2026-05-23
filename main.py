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
# Configuration
#
# Values here are defaults. Every key can be overridden at runtime by
# setting the matching ``VOICE_CLONE_*`` env var (see _env_overrides()
# below). For deployments via the Vocence /studio/ops fleet manager, the
# defaults below match the dashboard's expectations: port 8113, bearer
# auth via VOICE_CLONE_API_KEY, cap=1 concurrent request.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8113,
    "tts_model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "tts_language": "English",
    "tts_max_new_tokens": 2048,
    # Hard timeout per HTTP request for generation work.
    "request_timeout_seconds": 300,
    # Fixed-voice profiles used by /clone_const and other clone_* endpoints:
    "fixed_voice_dir": str(Path(__file__).resolve().parent / "bt_voices"),
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
    "const_auth_token": "logos_bot_token",
    # Hugging Face token for gated models; None → use HF_TOKEN / HUGGINGFACE_HUB_TOKEN env
    "hf_token": None,
    "preload_tts": False,
    # Vocence /studio/ops integration ---------------------------------------
    # Bearer token required on /healthz and /metrics. None / empty = open.
    # On the rented box, set VOICE_CLONE_API_KEY to match what the dashboard
    # registered the pod with.
    "api_key": None,
    # Max concurrent requests against the model. The single-thread engine
    # means cap > 1 just queues; cap=1 fails fast with 503 server_busy.
    "cap": 1,
}
# ---------------------------------------------------------------------------


def _env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Layer env vars on top of CONFIG. All env vars are prefixed
    ``VOICE_CLONE_`` so they don't collide with other services on the
    same host."""
    def _coerce(default, raw: str):
        if isinstance(default, bool):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int) and not isinstance(default, bool):
            try:
                return int(raw)
            except ValueError:
                return default
        if isinstance(default, float):
            try:
                return float(raw)
            except ValueError:
                return default
        return raw

    out = dict(cfg)
    # Special-cased env names — match the dashboard's deploy modal contract.
    pairs = [
        ("HOST", "host"),
        ("PORT", "port"),
        ("VOICE_CLONE_API_KEY", "api_key"),
        ("VOICE_CLONE_CAP", "cap"),
        ("VOICE_CLONE_REQUEST_TIMEOUT", "request_timeout_seconds"),
        ("VOICE_CLONE_HF_TOKEN", "hf_token"),
        ("VOICE_CLONE_PRELOAD", "preload_tts"),
    ]
    for env_key, cfg_key in pairs:
        raw = os.environ.get(env_key)
        if raw is not None and raw.strip() != "":
            out[cfg_key] = _coerce(cfg.get(cfg_key), raw)
    return out


CONFIG = _env_overrides(CONFIG)

import librosa
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
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


# ---------------------------------------------------------------------------
# Vocence /studio/ops integration: bearer auth, inflight tracking, /healthz,
# /metrics. Added on top of the existing endpoints with zero shape changes —
# legacy callers keep working; the Vocence dashboard's poller reads /healthz
# and /metrics for fleet visibility.
# ---------------------------------------------------------------------------
import collections as _collections
import time as _time
from fastapi.responses import JSONResponse as _JSONResponse


class _Metrics:
    def __init__(self, recent_window: int = 1000) -> None:
        self._lock = threading.Lock()
        self.start_ts = _time.time()
        self.requests_total = 0
        self.requests_ok = 0
        self.requests_err: dict[str, int] = {}
        self.duration_ms_sum = 0.0
        self.duration_ms_count = 0
        self.recent_durations_ms: "_collections.deque[float]" = _collections.deque(maxlen=recent_window)
        self.bytes_sent_total = 0
        self.audio_ms_total = 0  # not tracked here (would need to decode each WAV) — kept 0 for shape parity

    def record_success(self, duration_ms: float, bytes_sent: int = 0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_ok += 1
            self.duration_ms_sum += duration_ms
            self.duration_ms_count += 1
            self.recent_durations_ms.append(duration_ms)
            self.bytes_sent_total += bytes_sent

    def record_error(self, code: str, duration_ms: float = 0.0) -> None:
        with self._lock:
            self.requests_total += 1
            self.requests_err[code] = self.requests_err.get(code, 0) + 1
            if duration_ms > 0:
                self.duration_ms_sum += duration_ms
                self.duration_ms_count += 1
                self.recent_durations_ms.append(duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            durations = sorted(self.recent_durations_ms)
            n = len(durations)
            def pct(p: float) -> float:
                if n == 0:
                    return 0.0
                return durations[min(n - 1, int(p * n))]
            return {
                "uptime_seconds": int(_time.time() - self.start_ts),
                "requests_total": self.requests_total,
                "requests_ok": self.requests_ok,
                "requests_err": dict(self.requests_err),
                "duration_ms_sum": self.duration_ms_sum,
                "duration_ms_count": self.duration_ms_count,
                "duration_ms_avg": (self.duration_ms_sum / self.duration_ms_count) if self.duration_ms_count else 0.0,
                "duration_ms_p50": pct(0.50),
                "duration_ms_p95": pct(0.95),
                "duration_ms_p99": pct(0.99),
                "bytes_sent_total": self.bytes_sent_total,
                "audio_ms_total": self.audio_ms_total,
            }


_metrics = _Metrics()


class _InflightTracker:
    def __init__(self, cap: int) -> None:
        self._cap = max(1, int(cap))
        self._count = 0
        self._lock = asyncio.Lock()

    @property
    def cap(self) -> int: return self._cap

    @property
    def inflight(self) -> int: return self._count

    async def try_acquire(self) -> bool:
        async with self._lock:
            if self._count >= self._cap:
                return False
            self._count += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            self._count = max(0, self._count - 1)


_inflight = _InflightTracker(cap=int(CONFIG.get("cap", 1)))


# Endpoints that should NOT count toward inflight / metrics (they're the
# observability endpoints themselves + FastAPI's docs).
_OPS_PATHS = {"/healthz", "/metrics", "/health", "/docs", "/redoc", "/openapi.json"}


def _check_bearer(request) -> _JSONResponse | None:
    """Return a 401 JSONResponse if the configured API key is set and the
    request lacks the matching Authorization header. Returns None on
    success or when no key is configured (dev / unauthenticated mode)."""
    key = (CONFIG.get("api_key") or "").strip()
    if not key:
        return None
    header = request.headers.get("authorization", "")
    if header == f"Bearer {key}":
        return None
    return _JSONResponse(
        {"type": "error", "code": "auth", "message": "missing or invalid bearer token"},
        status_code=401,
    )


@app.middleware("http")
async def _ops_middleware(request, call_next):
    """Apply inflight tracking + metrics recording on every non-ops endpoint.
    Skips /healthz and /metrics so they're always reachable for monitoring,
    even when the model is saturated."""
    path = request.url.path
    if path in _OPS_PATHS:
        return await call_next(request)

    if not await _inflight.try_acquire():
        _metrics.record_error("server_busy", 0)
        return _JSONResponse(
            {"type": "error", "code": "server_busy", "message": f"inflight cap ({_inflight.cap}) exhausted"},
            status_code=503,
        )

    t0 = _time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = (_time.perf_counter() - t0) * 1000.0
        if response.status_code < 400:
            _metrics.record_success(elapsed)
        else:
            _metrics.record_error(f"http_{response.status_code}", elapsed)
        return response
    except Exception as e:
        _metrics.record_error(f"exception_{type(e).__name__}", (_time.perf_counter() - t0) * 1000.0)
        raise
    finally:
        await _inflight.release()


@app.get("/health")
async def health():
    # Legacy endpoint kept for back-compat (existing callers in the wild).
    return {"status": "ok"}


@app.get("/healthz")
async def healthz(request: Request):
    """Vocence ops dashboard health probe. Bearer-auth'd when api_key
    is configured. Returns the standardized shape the ops poller expects."""
    err = _check_bearer(request)
    if err is not None:
        return err
    return _JSONResponse({
        "status": "ok",
        "service": "voice_clone",
        "model_id": CONFIG.get("tts_model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        "sample_rate": 24000,
        "inflight": _inflight.inflight,
        "cap": _inflight.cap,
        "dev_stub": False,
    })


@app.get("/metrics")
async def metrics(request: Request):
    """Vocence ops dashboard scrape endpoint. Same JSON shape as the
    fast-tts-streaming + voice-design-non-streaming servers so the
    backend's poller treats every service uniformly."""
    err = _check_bearer(request)
    if err is not None:
        return err
    snap = _metrics.snapshot()
    snap["service"] = "voice_clone"
    snap["inflight"] = _inflight.inflight
    snap["cap"] = _inflight.cap
    return _JSONResponse(snap)


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
    except HTTPException:
        raise
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
    except HTTPException:
        raise
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
