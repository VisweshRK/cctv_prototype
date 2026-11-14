"""Verification Test for YAMNet ONNX Acoustic Audio Classifier.
Verifies model loading, 16 kHz stream classification, latency, and baby crying state machine.
"""

from datetime import datetime, timezone, timedelta
import os
import time
import numpy as np
import pytest

from ai.plugins.baby_crying.plugin import BabyCryingPlugin
from packages.event_schemas.events import StandardEventType, EventSeverity


def test_yamnet_model_loading_and_properties():
    plugin = BabyCryingPlugin(model_path="ai/models/yamnet.onnx")
    assert plugin.backend == "onnx_yamnet"
    assert plugin.session is not None
    assert plugin.input_name == "waveform"
    assert plugin.model_version == "yamnet_onnx_1.0.0"


def test_yamnet_inference_latency_under_50ms():
    plugin = BabyCryingPlugin(model_path="ai/models/yamnet.onnx")
    waveform = np.random.uniform(-0.1, 0.1, 16000).astype(np.float32)

    # Warmup
    plugin.session.run(None, {plugin.input_name: waveform})

    t0 = time.time()
    outputs = plugin.session.run(None, {plugin.input_name: waveform})
    elapsed_ms = (time.time() - t0) * 1000

    print(f"\nReal-Time YAMNet Audio Inference Latency: {elapsed_ms:.2f} ms")
    assert elapsed_ms < 50.0  # Real-time requirement
    assert outputs[0].shape[-1] == 521


def test_yamnet_crying_episode_state_machine_with_audio():
    plugin = BabyCryingPlugin(
        model_path="ai/models/yamnet.onnx",
        continuation_interval_sec=5.0,
        silence_timeout_sec=4.0
    )
    now = datetime.now(timezone.utc)

    # 1. Silence / low noise: Should NOT trigger crying
    silence = np.random.normal(0, 0.005, 16000).astype(np.float32)
    ev_silence = plugin.on_audio("cam1", silence, now)
    assert len(ev_silence) == 0

    # 2. Simulated infant crying audio: Test override or cry signal
    plugin.config["test_override_cry_score"] = 0.85
    cry_audio = np.random.uniform(-0.3, 0.3, 16000).astype(np.float32)
    ev_cry_start = plugin.on_audio("cam1", cry_audio, now)

    assert any(e.event_type == StandardEventType.BABY_CRY_STARTED.value for e in ev_cry_start)
    assert ev_cry_start[0].severity == EventSeverity.HIGH

    # 3. Ongoing cry after continuation interval
    t_cont = now + timedelta(seconds=6)
    ev_cry_cont = plugin.on_audio("cam1", cry_audio, t_cont)
    assert any(e.event_type == StandardEventType.BABY_CRY_CONTINUES.value for e in ev_cry_cont)

    # 4. Cry ended on silence timeout (feed silence across timeout window)
    plugin.config["test_override_cry_score"] = 0.0
    ev_cry_end = []
    for step in range(3):
        t_silence = t_cont + timedelta(seconds=step + 1)
        ev_cry_end.extend(plugin.on_audio("cam1", silence, t_silence))

    # After silence timeout (4.0s)
    t_timeout = t_cont + timedelta(seconds=6)
    ev_cry_end.extend(plugin.on_audio("cam1", silence, t_timeout))

    assert any(e.event_type == StandardEventType.BABY_CRY_ENDED.value for e in ev_cry_end)
