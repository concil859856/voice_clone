# Qwen3 voice cloning (TTS) workspace

This repository mixes a **single-file HTTP service** for voice cloning and an **optional multi-module API** that can call a remote **Qwen3-ASR** server for transcription. Both paths use **Qwen3-TTS** for synthesis.

---

## Does `main.py` cover the endpoint on its own?

**Yes.** Running `python main.py` starts a self-contained **FastAPI** server. You do **not** need the `clone/` package or vLLM on the same machine for this service.

- **Input:** `reference_audio` (file), `ref_text`, `target_text` (multipart form).
- **Output:** WAV bytes (`audio/wav`).

You must supply an accurate **`ref_text`** (what is spoken in the reference clip). The server runs **Qwen3-TTS** locally and loads the model on first request (or at startup if `preload_tts` is true in `CONFIG`).

**Dependencies (typical):** Python 3.10+, CUDA optional but recommended, `torch`, `qwen_tts`, `transformers`, `accelerate`, `librosa`, `soundfile`, `fastapi`, `uvicorn`, `python-multipart`. For gated Hugging Face models, set `HF_TOKEN` or put the token in `CONFIG["hf_token"]` inside `main.py`.

```bash
python main.py
```

- API docs: `http://127.0.0.1:8088/docs` (default host/port from `CONFIG`).
- Endpoint 1: **`POST http://<host>:<port>/voice-clone`** (uploaded reference audio + texts)
- Endpoint 2: **`POST http://<host>:<port>/clone_const`** (fixed reference voice from `const_reborn.wav`)

---

## Project layout

| Path | Role |
|------|------|
| **`main.py`** | **Primary HTTP service.** All important settings live in the `CONFIG` dict at the top. Exposes `POST /voice-clone` (WAV out), `POST /clone_const` (fixed reference voice; token-protected; WAV/OGG out), `GET /health`, and auto-generated `/docs`. |
| **`example_clone_const_client.py`** | Remote-client example for calling `/clone_const` with token auth and saving output locally. |
| **`test_voice_clone_local.py`** | Example client: POSTs a sample MP3 + form fields to the local server and saves `test_voice_clone_out.wav`. Run `main.py` first. |
| **`voice_clone.py`** | **CLI / script** version: loads TTS from disk paths, writes a WAV file. Useful for one-off experiments without HTTP. |
| **`test.py`** | Minimal inline test calling `Qwen3TTSModel` directly and writing `output_voice_clone.wav`. |
| **`clone/`** | **Separate FastAPI app** (`python -m clone` with `PYTHONPATH` set to the repo root). Includes **ASR over HTTP** (OpenAI-compatible `POST .../audio/transcriptions` on a **remote vLLM** host), JSON responses with base64 audio, and modular `app/services/*`. See below. |
| **`clone/requirements.txt`** | Extra deps for the `clone` app (`fastapi`, `uvicorn`, `httpx`, etc.). |
| **`clone/requirements-vllm-asr.txt`** | Audio decode deps (`librosa`, `soundfile`) intended for the **venv that runs `vllm serve`**, so `/v1/audio/transcriptions` works. |
| **Sample media** | e.g. `sample-000000.mp3`, `sample-000009.mp3` — local test assets (not required by the code paths; paths are configurable). |

---

## `clone/` package (optional, ASR + TTS)

Use this when you want callers to upload **only reference audio + target text** and have the backend **transcribe the reference** via **Qwen3-ASR on vLLM** before cloning.

- **Run:** from repo root, `PYTHONPATH=/path/to/repo python -m clone` (or `uvicorn clone.app.main:app ...`).
- **Configuration:** environment variables — see `clone/app/config.py` (e.g. `VLLM_ASR_BASE_URL`, `TTS_*`, `HF_TOKEN`, `CLONE_API_PORT`). Defaults may point at a remote ASR host.
- **`clone/app/main.py`** — FastAPI routes: `/api/v1/transcribe`, `/api/v1/voice-clone` (optional `reference_transcript`; if missing, calls ASR).
- **`clone/app/services/asr_client.py`** — HTTP client to the vLLM transcription endpoint.
- **`clone/app/services/tts_engine.py`** — Qwen3-TTS loading and `generate_voice_clone`.
- **`clone/app/schemas.py`** — Pydantic response models for JSON APIs.

The **root `main.py`** does **not** use vLLM; the **`clone/`** app **does** for automatic reference transcription when you omit the transcript.

---

## Quick comparison

| | `main.py` | `clone/` app |
|--|-----------|----------------|
| Voice clone | Yes | Yes |
| Config | `CONFIG` in `main.py` | Env vars (`clone/app/config.py`) |
| ASR / vLLM | No | Yes (remote URL) |
| Response | Raw WAV | JSON + base64 WAV (voice-clone route) |
| Start | `python main.py` | `PYTHONPATH=. python -m clone` |

---

## `.gitignore`

The repo ignores `venv/`, `__pycache__/`, `*.wav`, and `.env` so virtualenvs and generated audio stay local.

---

## `/clone_const` auth and usage

`/clone_const` uses simple token auth via request header:

- Header name: `X-Token`
- Required value: `vexor_bot_token`

Server-side token is configured in `main.py`:

```python
"const_auth_token": "vexor_bot_token"
```

Example curl:

```bash
curl -X POST "http://127.0.0.1:8088/clone_const" \
  -H "X-Token: vexor_bot_token" \
  -F "text=it's pretty impressive they are doing it well" \
  -F "output_format=ogg" \
  --output const_out.ogg
```

Or use `example_clone_const_client.py` from another server/application and save the returned bytes to a local file.
