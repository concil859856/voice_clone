#!/usr/bin/env python3
"""
Local test for the /voice-clone API.

1) Start the server:  python main.py
2) Run this script:    python test_voice_clone_local.py
"""
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8088"
AUDIO_PATH = Path("/workspace/sample-000009.mp3")
REF_TEXT = "the teacher thought that he'd taught himself all he could"
TARGET_TEXT = "it's pretty impressive they are doing it well"
OUTPUT_WAV = Path("/workspace/test_voice_clone_out.wav")


def main() -> None:
    if not AUDIO_PATH.is_file():
        raise SystemExit(f"Audio not found: {AUDIO_PATH}")

    url = f"{BASE_URL.rstrip('/')}/voice-clone"
    with AUDIO_PATH.open("rb") as f:
        files = {"reference_audio": (AUDIO_PATH.name, f, "audio/mpeg")}
        data = {"ref_text": REF_TEXT, "target_text": TARGET_TEXT}
        r = httpx.post(url, files=files, data=data, timeout=httpx.Timeout(600.0, connect=30.0))

    if r.status_code != 200:
        raise SystemExit(f"HTTP {r.status_code}: {r.text[:2000]}")

    OUTPUT_WAV.write_bytes(r.content)
    print(f"OK → {OUTPUT_WAV} ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()
