#!/usr/bin/env python3
"""
Example client for fixed-voice clone endpoints on a remote server.

Supported voices/endpoints:
  - const     -> /clone_const
  - assistant -> /clone_assistant
  - mark      -> /clone_mark
  - nova      -> /clone_nova
  - joker     -> /clone_joker

Usage:
  python example_clone_const_client.py
"""
from pathlib import Path

import httpx

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
BASE_URL = "http://149.36.0.184:8088"
TOKEN = "logos_bot_token"
TEXT = "it's pretty impressive they are doing it well"
OUTPUT_FORMAT = "ogg"  # "wav" or "ogg"
VOICE = "const"  # const | assistant | mark | nova | joker

VOICE_TO_ENDPOINT = {
    "const": "clone_const",
    "assistant": "clone_assistant",
    "mark": "clone_mark",
    "nova": "clone_nova",
    "joker": "clone_joker",
}


def main() -> None:
    voice_key = VOICE.strip().lower()
    endpoint = VOICE_TO_ENDPOINT.get(voice_key)
    if endpoint is None:
        valid = ", ".join(sorted(VOICE_TO_ENDPOINT))
        raise SystemExit(f"Invalid VOICE='{VOICE}'. Use one of: {valid}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{endpoint}_example.{OUTPUT_FORMAT}"
    url = f"{BASE_URL.rstrip('/')}/{endpoint}"
    headers = {"X-Token": TOKEN}
    data = {"text": TEXT, "output_format": OUTPUT_FORMAT}

    with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
        response = client.post(url, headers=headers, data=data)

    if response.status_code != 200:
        raise SystemExit(f"HTTP {response.status_code}: {response.text}")

    output_path.write_bytes(response.content)
    print(f"Saved: {output_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
