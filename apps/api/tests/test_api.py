"""API Integration and Domain Unit Tests for Presence B2C CCTV AI Platform.
Directly exercises route handlers, dynamic rule simulation, alerts,
super admin, and user capabilities.
"""

import uuid
import pytest
from apps.api.app.api.admin import get_system_health, list_all_households
from apps.api.app.api.alerts import acknowledge_alert, list_alerts
from apps.api.app.api.auth import login, register, LoginRequest, RegisterRequest
from apps.api.app.api.cameras import list_cameras, probe_camera_compatibility, CameraProbeRequest
from apps.api.app.api.events import simulate_event_ingestion, SimulateEventRequest, list_events
from apps.api.app.api.models import list_models
from apps.api.app.api.rules import list_rules, simulate_rule, RuleSimulateRequest
from apps.api.app.core.database import SessionLocal
from apps.api.app.core.init_db import init_database
from apps.api.app.models.entities import User


@pytest.fixture(scope="module", autouse=True)
def setup_database():
    init_database()


def test_auth_login_and_register():
    with SessionLocal() as db:
        # Test seeded demo user login
        res = login(LoginRequest(email="demo@presence.ai", password="demo123"), db=db)
        assert "access_token" in res
        assert res["user"]["email"] == "demo@presence.ai"

        # Test registration with unique email
        uid = uuid.uuid4().hex[:6]
        new_reg = register(RegisterRequest(
            email=f"resident_{uid}@presence.ai",
            password="testpass123",
            full_name="Ananya Verma",
            household_name="Verma Household"
        ), db=db)
        assert "access_token" in new_reg
        assert f"resident_{uid}@presence.ai" == new_reg["user"]["email"]


def test_camera_list_and_hardware_probe():
    with SessionLocal() as db:
        demo_user = db.query(User).filter_by(email="demo@presence.ai").first()
        cams = list_cameras(db=db, user=demo_user)
        assert len(cams) >= 1
        assert cams[0]["id"] == "camera_main"

        # Test non-destructive camera hardware compatibility probe
        report = probe_camera_compatibility(CameraProbeRequest(
            ip="192.168.1.100",
            rtsp_port=554,
            onvif_port=80
        ))
        assert "deployment_mode" in report
        assert "limitations" in report


def test_electricity_waste_rule_simulation():
    sample_rule = {
        "name": "Electricity Waste During Absence Alert",
        "conditions": {
            "operator": "AND",
            "items": [
                {"field": "event.type", "operator": "EQ", "value": "LIGHT_POSSIBLY_LEFT_ON"}
            ]
        },
        "actions": [
            {
                "type": "PUSH_NOTIFICATION",
                "recipient_group": "HOUSEHOLD_ADMINS",
                "custom_message_template": "Energy Alert: Lights left on while room is empty at {event.time}!",
                "priority": "NORMAL"
            }
        ],
        "camera_ids": ["camera_main"]
    }
    sample_event = {
        "event_type": "LIGHT_POSSIBLY_LEFT_ON",
        "camera_id": "camera_main",
        "zone_id": "zone_living_room",
        "metadata": {
            "estimated_waste_kwh": 0.45,
            "appliance_type": "light"
        }
    }
    sim_res = simulate_rule(RuleSimulateRequest(
        rule=sample_rule,
        sample_event=sample_event
    ))

    assert sim_res["rule_matches"] is True
    assert len(sim_res["simulated_notifications"]) >= 1
    assert "Energy Alert" in sim_res["simulated_notifications"][0]["message"]


def test_event_ingestion_and_alert_acknowledgement():
    with SessionLocal() as db:
        demo_user = db.query(User).filter_by(email="demo@presence.ai").first()

        # Ingest Critical AI event (e.g. Possible Dangerous Fire or Fall)
        ev_res = simulate_event_ingestion(SimulateEventRequest(
            module_id="fire",
            event_type="POSSIBLE_DANGEROUS_FIRE",
            camera_id="camera_main",
            severity="CRITICAL",
            confidence=0.96,
            metadata={
                "location": "Living Room Flare",
                "area": 1500
            }
        ), user=demo_user)

        assert ev_res["dispatched_alerts_count"] >= 1
        alert_id = ev_res["dispatched_alerts"][0]["alert_id"]

        # List alerts
        alerts = list_alerts(user=demo_user)
        assert any(a["alert_id"] == alert_id for a in alerts)

        # Acknowledge alert
        ack_res = acknowledge_alert(alert_id=alert_id, user=demo_user)
        assert ack_res["status"] == "acknowledged"


def test_model_registry_commercial_audit():
    models = list_models()
    assert len(models) >= 4
    for m in models:
        assert "license" in m
        assert "commercial_use_permission" in m
        assert "validation_status" in m


def test_super_admin_endpoints():
    with SessionLocal() as db:
        admin_user = db.query(User).filter_by(email="admin@presence.ai").first()

        # Health check
        health = get_system_health(user=admin_user)
        assert health["status"] == "HEALTHY"

        # All households
        households = list_all_households(db=db, user=admin_user)
        assert len(households) >= 2
