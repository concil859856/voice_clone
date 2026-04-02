import mimetypes

import httpx

from ..config import Settings


def _guess_mime(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


async def transcribe_audio_bytes(
    settings: Settings,
    audio_bytes: bytes,
    filename: str = "reference.wav",
) -> str:
    """
    Call a vLLM OpenAI-compatible ASR server (POST /v1/audio/transcriptions).
    Deploy with: vllm serve Qwen/Qwen3-ASR-1.7B
    """
    base = settings.vllm_asr_base_url.rstrip("/")
    url = f"{base}/audio/transcriptions"
    mime = _guess_mime(filename)
    headers = {"Authorization": f"Bearer {settings.vllm_api_key}"}

    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        resp = await client.post(
            url,
            headers=headers,
            files={"file": (filename, audio_bytes, mime)},
            data={"model": settings.vllm_asr_model},
        )
        resp.raise_for_status()
        payload = resp.json()

    text = payload.get("text")
    if not text:
        raise RuntimeError(f"ASR response missing 'text': {payload}")
    return str(text).strip()
