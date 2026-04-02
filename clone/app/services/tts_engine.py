from __future__ import annotations

import io
import threading
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig

from ..config import Settings


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
    """Decode arbitrary audio bytes to mono float32 + sample rate."""
    buf = io.BytesIO(data)
    try:
        wav, sr = sf.read(buf, dtype="float32", always_2d=False)
    except Exception:
        buf.seek(0)
        wav, sr = librosa.load(buf, sr=None, mono=True)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    return wav.astype(np.float32, copy=False), int(sr)


class TTSCloneEngine:
    """Lazy-loaded Qwen3-TTS voice cloning."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Optional[Qwen3TTSModel] = None

    def _load(self) -> Qwen3TTSModel:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:
                return self._model

            model_id = self._settings.tts_model_id
            hf_token = self._settings.hf_token

            if torch.cuda.is_available():
                device_map = "cuda:0"
                dtype = torch.bfloat16
            else:
                device_map = "cpu"
                dtype = torch.float32

            attn_impl = _pick_attn_implementation()
            AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
            model_config = AutoConfig.from_pretrained(model_id, token=hf_token)
            _set_config_dtype_tree(model_config, dtype)

            self._model = Qwen3TTSModel.from_pretrained(
                model_id,
                config=model_config,
                device_map=device_map,
                dtype=dtype,
                attn_implementation=attn_impl,
                token=hf_token,
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
        language: str,
    ) -> Tuple[bytes, int]:
        model = self._load()
        wav_np, sr = _load_audio_from_bytes(ref_audio_bytes, ref_filename)

        out_wavs, sample_rate = model.generate_voice_clone(
            text=target_text,
            language=language,
            ref_audio=(wav_np, sr),
            ref_text=ref_text,
            max_new_tokens=self._settings.tts_max_new_tokens,
        )

        out_buf = io.BytesIO()
        sf.write(out_buf, out_wavs[0], sample_rate, format="WAV", subtype="PCM_16")
        return out_buf.getvalue(), int(sample_rate)
