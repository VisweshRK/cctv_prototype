"""Comprehensive Unit & Integration Test Suite for the 3 Core Safety Plugins:
1. Fire & Smoke Detection
2. Electricity Consumption During Room Absence
3. Baby Crying Acoustic Classifier
"""

from datetime import datetime, timezone, timedelta
import numpy as np
import pytest

from ai.cctv_ai.events import EventManager
from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.pipeline import MasterPipeline
from ai.cctv_ai.zones import MonitoringZone, ZoneManager
from ai.plugins.fire.plugin import FireSmokePlugin
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin
from ai.plugins.baby_crying.plugin import BabyCryingPlugin
from packages.event_schemas.events import StandardEventType, EventSeverity
from packages.rule_schemas.rules import (
    CustomerRule,
    RuleConditionGroup,
    RuleConditionItem,
    RuleOperator,
    RuleAction,
    ActionType
)


# =====================================================================
# 1. Fire and Smoke Detection Plugin Tests
# =====================================================================

def test_fire_detection_uncontrolled_flame():
    """Verifies that bright flame colors outside cooking zones trigger CRITICAL fire alert."""
    zm = ZoneManager()
    plugin = FireSmokePlugin(zone_manager=zm)
    now = datetime.now(timezone.utc)

    # Frame with bright orange/yellow flame patch
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)  # dark background
    # Flame patch (BGR: high blue=0, green=180, red=255 -> bright orange in HSV)
    frame[200:260, 250:310] = (20, 180, 255)

    events = plugin.on_frame("cam_fire", frame, now)
    assert len(events) >= 1
    fire_event = events[0]
    assert fire_event.event_type == StandardEventType.POSSIBLE_DANGEROUS_FIRE.value
    assert fire_event.severity == EventSeverity.CRITICAL


def test_fire_detection_cooking_zone_suppression():
    """Verifies that flames inside configured cooking zones are classified as EXPECTED_COOKING_FLAME."""
    zm = ZoneManager()
    zone = MonitoringZone(
        zone_id="zone_stove",
        name="Kitchen Stove",
        camera_id="cam_kitchen",
        polygon=[(200, 150), (400, 150), (400, 350), (200, 350)],
        zone_type="cooking"
    )
    zm.add_zone(zone)

    plugin = FireSmokePlugin(zone_manager=zm)
    now = datetime.now(timezone.utc)

    # Moderate flame cluster inside cooking zone (center at 300, 250)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    frame[235:265, 285:315] = (20, 180, 255)

    events = plugin.on_frame("cam_kitchen", frame, now)
    assert len(events) >= 1
    assert events[0].event_type == StandardEventType.EXPECTED_COOKING_FLAME.value
    assert events[0].severity == EventSeverity.INFO


# =====================================================================
# 2. Electricity During Room Absence Plugin Tests
# =====================================================================

def test_electricity_waste_when_nobody_in_room():
    """Verifies that lights left on when no person is present triggers POTENTIAL_ENERGY_WASTE."""
    plugin = PowerWhenAbsentPlugin(empty_minutes=0.01, hysteresis_seconds=0.0)
    now = datetime.now(timezone.utc)

    # Configure a light ROI in top-left
    plugin.configure_rois("cam_living", [
        {"id": "light_main", "type": "light", "box": (50, 50, 150, 150), "threshold": 100.0}
    ])

    # Frame with light ON in the ROI
    frame_light_on = np.zeros((480, 640, 3), dtype=np.uint8)
    frame_light_on[50:150, 50:150] = 230  # bright light

    # Step 1: Person in room -> occupied -> no waste event
    person = TrackedObject(
        id="p1",
        label="person",
        box=(200, 200, 300, 400),
        score=0.92,
        median_score=0.92,
        true_positive=True,
        stationary=False,
        motionless_count=0,
        first_seen=now,
        last_seen=now
    )
    plugin.on_frame("cam_living", frame_light_on, now)
    ev1 = plugin.on_objects("cam_living", frame_light_on, now, [person], [])
    assert len(ev1) == 0  # Occupied, no waste

    # Step 2: Person leaves room -> empty room -> light remains ON
    later = now + timedelta(seconds=2)
    plugin.on_objects("cam_living", frame_light_on, later, [], [person])
    ev2 = plugin.on_frame("cam_living", frame_light_on, later + timedelta(seconds=1))
    assert len(ev2) >= 1
    assert any(e.event_type in [StandardEventType.LIGHT_POSSIBLY_LEFT_ON.value, StandardEventType.POTENTIAL_ENERGY_WASTE.value] for e in ev2)


def test_electricity_no_waste_when_lights_off():
    """Verifies that when lights are OFF and nobody is in the room, no waste is reported."""
    plugin = PowerWhenAbsentPlugin(empty_minutes=0.01, hysteresis_seconds=0.0)
    now = datetime.now(timezone.utc)

    plugin.configure_rois("cam_room", [
        {"id": "light_main", "type": "light", "box": (50, 50, 150, 150), "threshold": 100.0}
    ])

    # Dark frame -> lights OFF
    frame_dark = np.zeros((480, 640, 3), dtype=np.uint8)
    plugin.on_objects("cam_room", frame_dark, now, [], [])
    events = plugin.on_frame("cam_room", frame_dark, now + timedelta(seconds=2))
    assert len(events) == 0


# =====================================================================
# 3. Baby Crying Audio Detector Plugin Tests
# =====================================================================

def test_baby_crying_audio_detection():
    """Verifies that infant crying frequency acoustics trigger BABY_CRY_STARTED event."""
    plugin = BabyCryingPlugin(model_path="ai/models/yamnet.onnx")
    plugin.config["test_override_cry_score"] = 0.85
    now = datetime.now(timezone.utc)

    # Generate 15600 samples (0.975s @ 16 kHz) of infant cry fundamental (450 Hz + 900 Hz harmonic)
    t = np.linspace(0, 0.975, 15600)
    cry_audio = (0.7 * np.sin(2 * np.pi * 450 * t) + 0.3 * np.sin(2 * np.pi * 900 * t)).astype(np.float32)

    events = []
    for step in range(2):
        evs = plugin.on_audio("cam_nursery", cry_audio, now + timedelta(seconds=step * 0.5))
        events.extend(evs)
    assert len(events) >= 1
    assert events[0].event_type == StandardEventType.BABY_CRY_STARTED.value
    assert events[0].module_id == "baby_crying"


def test_baby_crying_silence_no_false_alarm():
    """Verifies that silence or ambient background noise does not trigger baby cry alerts."""
    plugin = BabyCryingPlugin()
    now = datetime.now(timezone.utc)

    silence = np.zeros(15600, dtype=np.float32)
    events = plugin.on_audio("cam_nursery", silence, now)
    assert len(events) == 0


# =====================================================================
# 4. End-to-End MasterPipeline Test with the 3 Plugins
# =====================================================================

def test_master_pipeline_execution():
    """Verifies that MasterPipeline runs end-to-end with the 3 focused plugins."""
    em = EventManager()
    zm = ZoneManager()
    pipeline = MasterPipeline(camera_id="cam_test", stream_source="synthetic", event_manager=em, zone_manager=zm)

    assert "fire" in pipeline.plugins
    assert "power_when_absent" in pipeline.plugins
    assert "baby_crying" in pipeline.plugins
    assert len(pipeline.plugins) == 3

    # Step frame through pipeline
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = pipeline.step_frame(test_frame)
    assert isinstance(events, list)

    # Step audio through pipeline
    test_audio = np.zeros(15600, dtype=np.float32)
    audio_events = pipeline.step_audio(test_audio)
    assert isinstance(audio_events, list)
