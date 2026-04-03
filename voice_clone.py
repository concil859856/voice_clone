#!/usr/bin/env python3
import os
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig

from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSConfig


def _set_config_dtype_tree(config: PretrainedConfig, dtype: torch.dtype) -> None:
    """
    Hugging Face sets config.dtype only on the root and immediate sub_configs.
    Qwen3-TTS nests talker -> code_predictor; leave that None and FlashAttention 2
    warns: 'without specifying a torch dtype'.
    """
    config.dtype = dtype
    for key in getattr(config, "sub_configs", None) or {}:
        child = getattr(config, key, None)
        if isinstance(child, PretrainedConfig):
            _set_config_dtype_tree(child, dtype)


def _pick_attn_implementation() -> str:
    """Prefer FlashAttention-2 on CUDA when installed; SDPA otherwise."""
    if not torch.cuda.is_available():
        return "sdpa"
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def main() -> None:
    model_id = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    text = "the teacher thought that he'd taught himself all he could"
    language = "English"
    ref_audio = "/workspace/sample-000009.mp3"
    ref_text = "the teacher thought that he'd taught himself all he could"
    output_path = "clone_out009.wav"
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    ref_audio_path = Path(ref_audio).expanduser().resolve()
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    if torch.cuda.is_available():
        device_map = "cuda:0"
        dtype = torch.bfloat16
    else:
        device_map = "cpu"
        dtype = torch.float32

    attn_impl = _pick_attn_implementation()
    print(f"Device: {device_map}, dtype: {dtype}, attention: {attn_impl}")

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    model_config = AutoConfig.from_pretrained(model_id, token=hf_token)
    _set_config_dtype_tree(model_config, dtype)

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        config=model_config,
        device_map=device_map,
        dtype=dtype,
        attn_implementation=attn_impl,
        token=hf_token,
    )

    # Default max_new_tokens in the library is 2048 codec steps; short lines need fewer.
    wavs, sample_rate = model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=str(ref_audio_path),
        ref_text=ref_text,
        max_new_tokens=1024,
    )
    sf.write(output_path, wavs[0], sample_rate)
    print(f"Saved cloned speech to: {output_path}")


if __name__ == "__main__":
    main()
