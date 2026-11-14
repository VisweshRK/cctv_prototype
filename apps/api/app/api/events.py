"""AI Events and Ingestion API Endpoints."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.database import SessionLocal
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import AIEventRecord, AlertRecord, User
from packages.event_schemas.events import StandardAIEvent, EventSeverity

router = APIRouter(prefix="/events", tags=["Events"])


class SimulateEventRequest(BaseModel):
    module_id: str
    event_type: str
    camera_id: str = "camera_main"
    severity: str = "HIGH"
    confidence: float = 0.92
    metadata: Dict[str, Any] = {}


@router.get("")
def list_events(
    module_id: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    user: User = Depends(get_current_user)
):
    """List events — queries SQLite DB first, falls back to in-memory deque."""
    lim = int(limit.default if hasattr(limit, "default") else limit)

    db = SessionLocal()
    try:
        q = db.query(AIEventRecord).order_by(AIEventRecord.timestamp.desc())
        if module_id:
            q = q.filter(AIEventRecord.module_id == module_id)
        if severity:
            q = q.filter(AIEventRecord.severity == severity)
        db_rows = q.limit(lim).all()
    finally:
        db.close()

    if db_rows:
        return [
            {
                "event_id": r.id,
                "household_id": r.household_id,
                "camera_id": r.camera_id,
                "module_id": r.module_id,
                "event_type": r.event_type,
                "severity": r.severity,
                "confidence": r.confidence,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "zone_id": r.zone_id,
                "metadata": r.metadata_json,
            }
            for r in db_rows
        ]

    # Fallback: in-memory deque (sorted newest-first)
    history = list(runtime.event_manager.event_history)
    filtered = []
    for ev in reversed(history):
        if module_id and ev.get("module_id") != module_id:
            continue
        if severity and ev.get("severity") != severity:
            continue
        filtered.append(ev)
        if len(filtered) >= lim:
            break
    return filtered


@router.post("/simulate")
def simulate_event_ingestion(
    req: SimulateEventRequest,
    user: User = Depends(get_current_user)
):
    """Manually ingests an AI event to test live notification flow and rules.
    Persists the event and any dispatched alerts to SQLite.
    """
    ev = StandardAIEvent(
        household_id=user.household_id or "household_default",
        camera_id=req.camera_id,
        module_id=req.module_id,
        event_type=req.event_type,
        confidence=req.confidence,
        severity=EventSeverity(req.severity) if req.severity in EventSeverity.__members__ else EventSeverity.INFO,
        metadata=req.metadata
    )
    dispatched = runtime.event_manager.process_event(ev, ignore_schedule=True)

    db = SessionLocal()
    try:
        # Persist the AI event
        event_record = AIEventRecord(
            id=ev.event_id,
            household_id=ev.household_id,
            camera_id=ev.camera_id,
            module_id=ev.module_id,
            event_type=ev.event_type,
            timestamp=ev.timestamp if isinstance(ev.timestamp, datetime) else datetime.now(timezone.utc),
            confidence=ev.confidence,
            severity=ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity),
            zone_id=getattr(ev, "zone_id", None),
            metadata_json=ev.metadata or {},
        )
        db.add(event_record)

        # Persist each dispatched alert
        for alert in dispatched:
            alert_record = AlertRecord(
                id=alert.alert_id,
                household_id=alert.household_id,
                rule_id=alert.rule_id,
                rule_name=alert.rule_name,
                event_type=alert.event_type,
                message=alert.message,
                priority=alert.priority,
                camera_id=alert.camera_id,
                timestamp=alert.timestamp if isinstance(alert.timestamp, datetime) else datetime.now(timezone.utc),
                recipient_group=alert.recipient_group,
                metadata_json=alert.metadata or {},
                acknowledged=alert.acknowledged,
            )
            db.add(alert_record)

        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "event_id": ev.event_id,
        "dispatched_alerts_count": len(dispatched),
        "dispatched_alerts": [d.to_dict() for d in dispatched]
    }
