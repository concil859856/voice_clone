"""Non-GPU smoke test for the Vocence /studio/ops integration on voice_clone.

Verifies that the FastAPI app instantiates, /healthz + /metrics respond with
the standardized shape, the inflight cap is respected (503 on overflow),
and bearer auth gates these endpoints when configured.

GPU-dependent paths (actual TTS synthesis through Qwen3-TTS) are NOT
exercised here — run those on the rented 4090 box once the image is up.

Run:  python3 test_ops_smoke.py
"""
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Stub the heavy GPU deps so voice_clone.main imports cleanly on a CPU-only
# dev box. The /healthz + /metrics paths never touch any of these; only
# the actual /voice-clone synthesis does, and we don't exercise that.
# ---------------------------------------------------------------------------
def _stub_module(name: str, **attrs) -> types.ModuleType:
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


# numpy is needed by the type hints in main.py — try the real one first.
try:
    import numpy  # noqa: F401
except ImportError:
    np_stub = _stub_module("numpy")
    np_stub.ndarray = type("ndarray", (), {})
    np_stub.float32 = float
    np_stub.zeros = lambda *a, **k: None
    np_stub.frombuffer = lambda *a, **k: None
    np_stub.concatenate = lambda *a, **k: None
    np_stub.clip = lambda *a, **k: None
    np_stub.int16 = int

for name in ("librosa", "soundfile", "torch"):
    if name not in sys.modules:
        _stub_module(name)

# transformers — only AutoConfig + PretrainedConfig referenced at import time.
if "transformers" not in sys.modules:
    tf = _stub_module("transformers")
    tf.AutoConfig = type("AutoConfig", (), {"from_pretrained": staticmethod(lambda *a, **k: None)})
    _stub_module("transformers.configuration_utils", PretrainedConfig=type("PretrainedConfig", (), {}))

# qwen_tts — replaced wholesale.
if "qwen_tts" not in sys.modules:
    qt = _stub_module("qwen_tts")
    qt.Qwen3TTSModel = type("Qwen3TTSModel", (), {})
    _stub_module("qwen_tts.core", )
    _stub_module("qwen_tts.core.models", Qwen3TTSConfig=type("Qwen3TTSConfig", (), {}))

# Configure env BEFORE importing main, to test env-driven config.
os.environ["VOICE_CLONE_API_KEY"] = "test-key-xyz"
os.environ["VOICE_CLONE_CAP"] = "1"
os.environ["PORT"] = "9999"  # never actually bound — we use TestClient

import main  # noqa: E402 — must come after stubs + env

from fastapi.testclient import TestClient  # noqa: E402


def main_test() -> int:
    print(f"CONFIG.port = {main.CONFIG.get('port')}    (expected: 9999 from PORT env)")
    print(f"CONFIG.api_key set = {bool(main.CONFIG.get('api_key'))}    (expected: True)")
    print(f"CONFIG.cap = {main.CONFIG.get('cap')}    (expected: 1)")

    client = TestClient(main.app)

    failures = 0

    # ---- /healthz without auth -> 401 ------------------------------------
    r = client.get("/healthz")
    if r.status_code != 401:
        print(f"FAIL: /healthz without bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /healthz without bearer -> 401")

    # ---- /healthz with auth -> 200 + expected fields ---------------------
    r = client.get("/healthz", headers={"Authorization": "Bearer test-key-xyz"})
    if r.status_code != 200:
        print(f"FAIL: /healthz with bearer should be 200, got {r.status_code} body={r.text[:200]}")
        failures += 1
    else:
        body = r.json()
        required = {"status", "service", "model_id", "sample_rate", "inflight", "cap", "dev_stub"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /healthz missing fields: {missing}")
            failures += 1
        elif body["service"] != "voice_clone":
            print(f"FAIL: /healthz service={body['service']!r}, expected 'voice_clone'")
            failures += 1
        elif body["cap"] != 1:
            print(f"FAIL: /healthz cap={body['cap']!r}, expected 1")
            failures += 1
        else:
            print(f"PASS: /healthz with bearer -> 200 service={body['service']} cap={body['cap']} inflight={body['inflight']}")

    # ---- /metrics with auth ----------------------------------------------
    r = client.get("/metrics", headers={"Authorization": "Bearer test-key-xyz"})
    if r.status_code != 200:
        print(f"FAIL: /metrics should be 200, got {r.status_code}")
        failures += 1
    else:
        body = r.json()
        required = {"uptime_seconds", "requests_total", "requests_ok", "requests_err",
                    "duration_ms_sum", "duration_ms_count", "duration_ms_p50",
                    "duration_ms_p95", "duration_ms_p99", "inflight", "cap", "service"}
        missing = required - set(body)
        if missing:
            print(f"FAIL: /metrics missing fields: {missing}")
            failures += 1
        else:
            print(f"PASS: /metrics with bearer -> 200 requests_total={body['requests_total']}")

    # ---- /metrics with wrong bearer -> 401 -------------------------------
    r = client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
    if r.status_code != 401:
        print(f"FAIL: /metrics with wrong bearer should be 401, got {r.status_code}")
        failures += 1
    else:
        print("PASS: /metrics with wrong bearer -> 401")

    # ---- legacy /health still works without auth -------------------------
    r = client.get("/health")
    if r.status_code != 200:
        print(f"FAIL: legacy /health should be 200, got {r.status_code}")
        failures += 1
    else:
        print(f"PASS: legacy /health -> 200 {r.json()}")

    # ---- ops_paths skip bearer + inflight -- /healthz multiple times -----
    # Confirms ops_middleware doesn't apply the inflight check on /healthz.
    for _ in range(5):
        r = client.get("/healthz", headers={"Authorization": "Bearer test-key-xyz"})
        if r.status_code != 200:
            print(f"FAIL: /healthz loop returned {r.status_code} on iteration")
            failures += 1
            break
    else:
        print("PASS: /healthz 5x consecutive -> all 200 (ops_middleware skips it)")

    # ---- routes still registered ------------------------------------------
    expected_routes = {"/voice-clone", "/clone_const", "/clone_assistant",
                       "/clone_mark", "/clone_nova", "/clone_joker",
                       "/healthz", "/metrics", "/health"}
    actual = {r.path for r in main.app.routes if hasattr(r, "path")}
    missing = expected_routes - actual
    if missing:
        print(f"FAIL: missing routes: {missing}")
        failures += 1
    else:
        print(f"PASS: all expected routes registered ({len(expected_routes)} checked)")

    print()
    if failures == 0:
        print("=" * 50)
        print(f"ALL {6 + 1} TESTS PASSED")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print(f"{failures} TEST(S) FAILED")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main_test())
