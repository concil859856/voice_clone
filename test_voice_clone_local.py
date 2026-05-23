#!/usr/bin/env python3
"""
Local endpoint tester for this repo.

Root app:
  1) python main.py
  2) python test_voice_clone_local.py health
  3) python test_voice_clone_local.py voice-clone --audio /path/ref.wav --ref-text "words in clip"
  4) python test_voice_clone_local.py fixed --voice const --text "hello"

Modular clone app:
  1) PYTHONPATH=. python -m clone
  2) python test_voice_clone_local.py clone-voice --audio /path/ref.wav --ref-text "words in clip"
"""
from __future__ import annotations

import argparse
import base64
import mimetypes
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_BASE_URL = "http://127.0.0.1:53211"
DEFAULT_TOKEN = "logos_bot_token"
DEFAULT_REF_TEXT = "the teacher thought that he'd taught himself all he could"
DEFAULT_TARGET_TEXT = "it's pretty impressive they are doing it well"
DEFAULT_OUTPUT = OUTPUT_DIR / "test_voice_clone_out.wav"
FIXED_VOICES = ("const", "assistant", "mark", "nova", "joker")


def _content_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _check_response(response: httpx.Response, label: str) -> None:
    if response.status_code != 200:
        raise SystemExit(f"{label} failed: HTTP {response.status_code}: {response.text[:2000]}")


def _write_output(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _require_audio(path: Path | None) -> Path:
    if path is None:
        raise SystemExit("Missing --audio. Pass a reference audio file path.")
    if not path.is_file():
        raise SystemExit(
            f"Audio not found: {path}\n"
            "Use an existing .wav/.mp3/.ogg/.m4a/.flac/.webm file, for example:\n"
            '  python test_voice_clone_local.py voice-clone --audio /workspace/my_voice.wav --ref-text "exact words"'
        )
    return path


def test_health(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url.rstrip('/')}/health")
    _check_response(response, "Health check")
    print(f"Health OK: {response.json()}")


def test_voice_clone(
    client: httpx.Client,
    base_url: str,
    audio_path: Path,
    ref_text: str,
    target_text: str,
    output_path: Path,
) -> None:
    with audio_path.open("rb") as audio:
        response = client.post(
            f"{base_url.rstrip('/')}/voice-clone",
            files={"reference_audio": (audio_path.name, audio, _content_type(audio_path))},
            data={"ref_text": ref_text, "target_text": target_text},
        )
    _check_response(response, "Voice clone")
    _write_output(output_path, response.content)
    print(f"Voice clone OK: wrote {output_path} ({len(response.content)} bytes)")


def test_fixed_voice(
    client: httpx.Client,
    base_url: str,
    voice: str,
    text: str,
    token: str,
    output_format: str,
    output_path: Path | None,
) -> None:
    endpoint = f"clone_{voice}"
    suffix = output_format.lower()
    out = output_path or OUTPUT_DIR / f"{endpoint}_out.{suffix}"
    response = client.post(
        f"{base_url.rstrip('/')}/{endpoint}",
        headers={"X-Token": token},
        data={"text": text, "output_format": output_format},
    )
    _check_response(response, endpoint)
    _write_output(out, response.content)
    print(f"{endpoint} OK: wrote {out} ({len(response.content)} bytes)")


def test_clone_api_voice(
    client: httpx.Client,
    base_url: str,
    audio_path: Path,
    ref_text: str,
    target_text: str,
    output_path: Path,
) -> None:
    with audio_path.open("rb") as audio:
        response = client.post(
            f"{base_url.rstrip('/')}/api/v1/voice-clone",
            files={"reference_audio": (audio_path.name, audio, _content_type(audio_path))},
            data={"reference_transcript": ref_text, "text": target_text},
        )
    _check_response(response, "Clone API voice clone")
    payload = response.json()
    _write_output(output_path, base64.standard_b64decode(payload["audio_base64"]))
    print(f"Clone API voice clone OK: wrote {output_path}")
    print(f"Transcript source: {payload.get('transcript_source')}")


def test_clone_api_transcribe(client: httpx.Client, base_url: str, audio_path: Path) -> None:
    with audio_path.open("rb") as audio:
        response = client.post(
            f"{base_url.rstrip('/')}/api/v1/transcribe",
            files={"reference_audio": (audio_path.name, audio, _content_type(audio_path))},
        )
    _check_response(response, "Clone API transcribe")
    print(f"Transcribe OK: {response.json().get('text')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test local voice clone endpoints.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Server base URL.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="Test GET /health.")

    voice_clone = subparsers.add_parser("voice-clone", help="Test root POST /voice-clone.")
    voice_clone.add_argument("--audio", type=Path, required=True, help="Reference audio file.")
    voice_clone.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Exact words in the reference audio.")
    voice_clone.add_argument("--target-text", default=DEFAULT_TARGET_TEXT, help="Text to synthesize.")
    voice_clone.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output audio path.")

    fixed = subparsers.add_parser("fixed", help="Test root fixed-voice endpoints.")
    fixed.add_argument("--voice", choices=FIXED_VOICES, default="const", help="Fixed voice to use.")
    fixed.add_argument("--text", default=DEFAULT_TARGET_TEXT, help="Text to synthesize.")
    fixed.add_argument("--token", default=DEFAULT_TOKEN, help="X-Token auth value.")
    fixed.add_argument("--output-format", choices=("wav", "ogg"), default="wav", help="Response format.")
    fixed.add_argument("--output", type=Path, help="Output audio path.")

    clone_voice = subparsers.add_parser("clone-voice", help="Test modular POST /api/v1/voice-clone.")
    clone_voice.add_argument("--audio", type=Path, required=True, help="Reference audio file.")
    clone_voice.add_argument("--ref-text", default=DEFAULT_REF_TEXT, help="Exact words in the reference audio.")
    clone_voice.add_argument("--target-text", default=DEFAULT_TARGET_TEXT, help="Text to synthesize.")
    clone_voice.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output WAV path.")

    clone_transcribe = subparsers.add_parser("clone-transcribe", help="Test modular POST /api/v1/transcribe.")
    clone_transcribe.add_argument("--audio", type=Path, required=True, help="Reference audio file.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeout = httpx.Timeout(600.0, connect=30.0)

    with httpx.Client(timeout=timeout) as client:
        if args.command == "health":
            test_health(client, args.base_url)
        elif args.command == "voice-clone":
            test_health(client, args.base_url)
            test_voice_clone(
                client,
                args.base_url,
                _require_audio(args.audio),
                args.ref_text,
                args.target_text,
                args.output,
            )
        elif args.command == "fixed":
            test_health(client, args.base_url)
            test_fixed_voice(
                client,
                args.base_url,
                args.voice,
                args.text,
                args.token,
                args.output_format,
                args.output,
            )
        elif args.command == "clone-voice":
            test_health(client, args.base_url)
            test_clone_api_voice(
                client,
                args.base_url,
                _require_audio(args.audio),
                args.ref_text,
                args.target_text,
                args.output,
            )
        elif args.command == "clone-transcribe":
            test_health(client, args.base_url)
            test_clone_api_transcribe(client, args.base_url, _require_audio(args.audio))


if __name__ == "__main__":
    main()
