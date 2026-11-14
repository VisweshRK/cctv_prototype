"""Dedicated Test Suite for Plan P1 Mohanraj: Electricity Used When Nobody Is There.
Tests Use Case 3 (ai/plugins/power_when_absent) according to Section 3 specifications:
1. Shared tracker stationary person tracking (person sitting still != empty).
2. Lights on calibration (brightness closer to 'on' vs 'off').
3. Q-Tech Night mode IR saturation check (grey image = lights off).
4. Screen on and fan moving frame-difference detection.
5. 60s absence departure hysteresis.
6. Timer per camera (N empty minutes -> event 'power_wasted').
7. Re-arm when person enters or all appliances turn off.
"""

from datetime import datetime, timezone, timedelta
import cv2
import numpy as np
import pytest

from ai.cctv_ai.objects import TrackedObject
from ai.cctv_ai.testing import (
    create_test_person,
    create_stationary_person,
    generate_test_frame,
    generate_ir_test_frame,
    generate_fan_frame_pair,
)
from ai.cctv_ai.zones import MonitoringZone, ZoneManager
from ai.plugins.power_when_absent.plugin import PowerWhenAbsentPlugin
from packages.event_schemas.events import StandardEventType


def test_stationary_person_does_not_trigger_waste():
    """Done when: 'no event while someone sits still'
    A person sitting still is still tracked (stationary=True), so the room isn't empty.
    """
    plugin = PowerWhenAbsentPlugin(empty_minutes=10.0, hysteresis_seconds=60.0)
    plugin.configure_rois("cam_living", [
        {"id": "light_main", "type": "light", "box": (50, 50, 150, 150), "threshold": 120.0}
    ])

    start_time = datetime(2026, 10, 30, 12, 0, 0, tzinfo=timezone.utc)
    # Bright frame with light on
    frame = generate_test_frame(light_on=True, light_roi=(50, 50, 150, 150))

    # Person enters and sits still for 15 minutes (motionless_count increments)
    for m in range(16):
        t = start_time + timedelta(minutes=m)
        person_sitting = create_stationary_person(
            track_id="couch_reader",
            box=(180, 150, 260, 320),
            timestamp=t,
            motionless_count=m * 10
        )
        assert person_sitting.stationary is True

        # Process objects and frame
        plugin.on_objects("cam_living", frame, t, [person_sitting], [])
        events = plugin.on_frame("cam_living", frame, t)

        # No event must ever be emitted while someone sits still in the room
        assert len(events) == 0, f"False alarm triggered at minute {m} while person sitting still!"

    cam_state = plugin._get_camera_state("cam_living")
    assert cam_state["is_empty"] is False
    assert cam_state["empty_since"] is None


def test_hysteresis_60s_departure():
    """Step 5: Hysteresis: 'empty' only after 60 s with no person, so someone leaving
    for a moment (e.g. 30s) does not start the timer.
    """
    plugin = PowerWhenAbsentPlugin(empty_minutes=5.0, hysteresis_seconds=60.0)
    plugin.configure_rois("cam1", [
        {"id": "light_1", "type": "light", "box": (50, 50, 150, 150), "threshold": 100.0}
    ])

    t0 = datetime(2026, 10, 30, 14, 0, 0, tzinfo=timezone.utc)
    person = create_test_person(track_id="p1", timestamp=t0)
    plugin.on_objects("cam1", None, t0, [person], [])
    assert plugin._get_camera_state("cam1")["is_empty"] is False

    # Person leaves room for 30 seconds (e.g. grabbing water)
    t_30s = t0 + timedelta(seconds=30)
    plugin.on_objects("cam1", None, t_30s, [], [])
    # Should NOT be confirmed empty yet
    assert plugin._get_camera_state("cam1")["is_empty"] is False
    assert plugin._get_camera_state("cam1")["empty_since"] is None

    # At 59 seconds: still within hysteresis
    t_59s = t0 + timedelta(seconds=59)
    plugin.on_objects("cam1", None, t_59s, [], [])
    assert plugin._get_camera_state("cam1")["is_empty"] is False

    # At 61 seconds: confirmed empty!
    t_61s = t0 + timedelta(seconds=61)
    plugin.on_objects("cam1", None, t_61s, [], [])
    assert plugin._get_camera_state("cam1")["is_empty"] is True
    assert plugin._get_camera_state("cam1")["empty_since"] is not None


def test_power_wasted_event_and_data_payload():
    """Step 4: Timer per camera: empty and power on for N min -> one event
    label: 'power_wasted', data: {'lights': true, 'screen': false, 'empty_minutes': 10}
    """
    plugin = PowerWhenAbsentPlugin(empty_minutes=10.0, hysteresis_seconds=60.0)
    plugin.configure_rois("cam_office", [
        {"id": "desk_lamp", "type": "light", "box": (40, 40, 120, 120), "threshold": 100.0},
        {"id": "work_monitor", "type": "screen", "box": (200, 100, 350, 250), "threshold": 80.0}
    ])

    t0 = datetime(2026, 10, 30, 9, 0, 0, tzinfo=timezone.utc)
    p = create_test_person(timestamp=t0)
    plugin.on_objects("cam_office", None, t0, [p], [])

    # Person leaves at t0. Frame has light ON, screen OFF
    frame = generate_test_frame(
        light_on=True,
        light_roi=(40, 40, 120, 120),
        screen_on=False,
        screen_roi=(200, 100, 350, 250)
    )

    # 1 minute later: empty confirmed
    t_1m = t0 + timedelta(minutes=1)
    plugin.on_objects("cam_office", frame, t_1m, [], [])
    events_1m = plugin.on_frame("cam_office", frame, t_1m)
    assert len(events_1m) == 0  # not yet 10 minutes

    # 11 minutes later: (10 minutes of confirmed absence)
    t_11m = t0 + timedelta(minutes=11)
    plugin.on_objects("cam_office", frame, t_11m, [], [])
    events_11m = plugin.on_frame("cam_office", frame, t_11m)

    assert len(events_11m) >= 1
    # Find main power_wasted event
    wasted_event = [e for e in events_11m if e.metadata.get("label") == "power_wasted"][0]
    assert wasted_event.event_type == StandardEventType.POTENTIAL_ENERGY_WASTE.value
    data = wasted_event.metadata["data"]
    assert data["lights"] is True
    assert data["screen"] is False
    assert data["empty_minutes"] >= 10.0

    # Ensure deduplication: next frame should NOT repeat event (one event per incident)
    t_12m = t0 + timedelta(minutes=12)
    events_12m = plugin.on_frame("cam_office", frame, t_12m)
    assert len(events_12m) == 0


def test_qtech_night_mode_ir_saturation():
    """Step 2: Night mode: when the Q-Tech camera switches to IR, the image turns grey
    (saturation ≈ 0). A grey IR image means the room lights are off.
    """
    plugin = PowerWhenAbsentPlugin(
        empty_minutes=5.0,
        hysteresis_seconds=0.0,
        check_ir_night_mode=True,
        ir_saturation_threshold=15.0
    )
    plugin.configure_rois("cam_qtech", [
        {"id": "chandelier", "type": "light", "box": (50, 50, 150, 150), "threshold": 120.0}
    ])

    t_empty = datetime(2026, 10, 30, 23, 0, 0, tzinfo=timezone.utc)
    plugin.update_occupancy_state("cam_qtech", is_occupied=False, timestamp=t_empty)

    # Room is dark at night, IR LEDs reflect off fixture causing brightness,
    # but saturation is 0 (monochrome / night vision)
    ir_frame = generate_ir_test_frame(
        light_roi=(50, 50, 150, 150),
        ir_reflection_brightness=190  # Bright reflection
    )

    t_check = t_empty + timedelta(minutes=10)
    events = plugin.on_frame("cam_qtech", ir_frame, t_check)

    # Must NOT alert because IR night mode indicates room lights are actually OFF
    assert len(events) == 0
    state = plugin._get_camera_state("cam_qtech")
    assert state["is_ir_night_mode"] is True
    assert state["power_signals"]["lights"] is False


def test_light_calibration_distance_to_on_off():
    """Step 2: At setup, record the average brightness with lights off and on.
    Lights are on when the ROI brightness is closer to the 'on' value.
    """
    plugin = PowerWhenAbsentPlugin(empty_minutes=2.0, hysteresis_seconds=0.0)
    # Calibrated: off = 40, on = 200 (midpoint = 120)
    plugin.configure_rois("cam1", [
        {
            "id": "calibrated_light",
            "type": "light",
            "box": (50, 50, 150, 150),
            "brightness_off": 40.0,
            "brightness_on": 200.0,
        }
    ])

    t0 = datetime(2026, 10, 30, 16, 0, 0, tzinfo=timezone.utc)
    plugin.update_occupancy_state("cam1", is_occupied=False, timestamp=t0)

    # Case A: ROI brightness = 80 (closer to 40 than 200 -> lights are OFF)
    frame_off = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame_off[50:150, 50:150] = (70, 80, 90)  # Mean ~80
    events_off = plugin.on_frame("cam1", frame_off, t0 + timedelta(minutes=5))
    assert len(events_off) == 0

    # Case B: ROI brightness = 160 (closer to 200 than 40 -> lights are ON)
    frame_on = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame_on[50:150, 50:150] = (150, 160, 170)  # Mean ~160
    events_on = plugin.on_frame("cam1", frame_on, t0 + timedelta(minutes=5))
    assert len(events_on) >= 1
    assert any(e.metadata.get("label") == "power_wasted" for e in events_on)


def test_fan_motion_frame_differencing():
    """Step 3: 'Fan on': constant motion inside a fan ROI (use frame differences
    inside the ROI only).
    """
    plugin = PowerWhenAbsentPlugin(empty_minutes=1.0, hysteresis_seconds=0.0)
    fan_box = (250, 20, 350, 100)
    plugin.configure_rois("cam1", [
        {"id": "fan_roi", "type": "fan", "box": fan_box, "motion_diff_threshold": 4.0}
    ])

    t0 = datetime(2026, 10, 30, 10, 0, 0, tzinfo=timezone.utc)
    plugin.update_occupancy_state("cam1", is_occupied=False, timestamp=t0)

    # Test moving fan: pair of frames with rotating fan blade
    f1, f2 = generate_fan_frame_pair(fan_roi=fan_box, fan_moving=True)
    plugin.on_frame("cam1", f1, t0 + timedelta(seconds=10))
    events = plugin.on_frame("cam1", f2, t0 + timedelta(minutes=2))

    assert len(events) >= 1
    wasted = [e for e in events if e.metadata.get("label") == "power_wasted"][0]
    assert wasted.metadata["data"]["fan"] is True


def test_rearming_when_person_returns():
    """Step 4: Re-arm when someone enters or everything turns off."""
    plugin = PowerWhenAbsentPlugin(empty_minutes=2.0, hysteresis_seconds=0.0)
    plugin.configure_rois("cam1", [
        {"id": "light1", "type": "light", "box": (50, 50, 150, 150), "threshold": 120.0}
    ])

    frame = generate_test_frame(light_on=True, light_roi=(50, 50, 150, 150))
    t0 = datetime(2026, 10, 30, 8, 0, 0, tzinfo=timezone.utc)
    plugin.update_occupancy_state("cam1", is_occupied=False, timestamp=t0)

    # Episode 1 alert
    events_1 = plugin.on_frame("cam1", frame, t0 + timedelta(minutes=3))
    assert len(events_1) >= 1
    assert plugin._get_camera_state("cam1")["alerted"] is True

    # Person re-enters the room!
    t_return = t0 + timedelta(minutes=5)
    person = create_test_person(timestamp=t_return)
    plugin.on_objects("cam1", frame, t_return, [person], [])

    # State must be re-armed
    assert plugin._get_camera_state("cam1")["alerted"] is False
    assert plugin._get_camera_state("cam1")["is_empty"] is False

    # Person leaves again
    t_leave = t0 + timedelta(minutes=10)
    plugin.on_objects("cam1", frame, t_leave, [], [])

    # Episode 2 alert after another 3 minutes of empty absence
    events_2 = plugin.on_frame("cam1", frame, t_leave + timedelta(minutes=3))
    assert len(events_2) >= 1
    assert plugin._get_camera_state("cam1")["alerted"] is True


def test_zone_containment_filtering():
    """Step 1: Use the zone helper if only part of the frame is the room."""
    zm = ZoneManager()
    living_room = MonitoringZone(
        zone_id="zone_room",
        name="Living Room",
        camera_id="cam_main",
        polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
        zone_type="room"
    )
    zm.add_zone(living_room)

    plugin = PowerWhenAbsentPlugin(
        empty_minutes=5.0,
        hysteresis_seconds=0.0,
        zone_manager=zm
    )
    plugin.configure_rois("cam_main", [
        {"id": "room_light", "type": "light", "box": (120, 120, 200, 200), "threshold": 100.0}
    ])

    t0 = datetime(2026, 10, 30, 11, 0, 0, tzinfo=timezone.utc)
    # Person outside the room polygon (e.g. outside window or garden path at 550, 450)
    person_outside = create_test_person(
        track_id="passerby",
        box=(500, 400, 600, 480),
        timestamp=t0
    )

    frame = generate_test_frame(light_on=True, light_roi=(120, 120, 200, 200))

    plugin.on_objects("cam_main", frame, t0, [person_outside], [])
    # Since person is outside room polygon, living room is considered empty!
    assert plugin._get_camera_state("cam_main")["is_empty"] is True

    # After 6 minutes, light is on and room is empty -> event triggered
    events = plugin.on_frame("cam_main", frame, t0 + timedelta(minutes=6))
    assert len(events) >= 1
