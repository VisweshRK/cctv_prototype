"""Alert Center API Endpoints: In-app notification center, sirens, acknowledgements."""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.database import SessionLocal
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import AlertRecord, User

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("")
def list_alerts(
    unacknowledged_only: bool = False,
    limit: int = 50,
    user: User = Depends(get_current_user)
):
    """List alerts — queries SQLite DB first (newest first, capped at 100),
    falls back to in-memory deque when DB is empty.
    """
    lim = int(limit.default if hasattr(limit, "default") else limit)
    cap = min(lim, 100)

    db = SessionLocal()
    try:
        q = db.query(AlertRecord).order_by(AlertRecord.timestamp.desc())
        if unacknowledged_only:
            q = q.filter(AlertRecord.acknowledged == False)  # noqa: E712
        db_rows = q.limit(cap).all()
    finally:
        db.close()

    if db_rows:
        return [
            {
                "alert_id": r.id,
                "household_id": r.household_id,
                "rule_id": r.rule_id,
                "rule_name": r.rule_name,
                "event_type": r.event_type,
                "message": r.message,
                "priority": r.priority,
                "camera_id": r.camera_id,
                "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                "recipient_group": r.recipient_group,
                "metadata": r.metadata_json,
                "acknowledged": r.acknowledged,
            }
            for r in db_rows
        ]

    # Fallback: in-memory deque (sorted newest-first)
    alerts = list(runtime.event_manager.alert_history)
    filtered = []
    for a in reversed(alerts):
        if unacknowledged_only and a.get("acknowledged", False):
            continue
        filtered.append(a)
        if len(filtered) >= cap:
            break
    return filtered


@router.post("/{alert_id}/ack")
def acknowledge_alert(
    alert_id: str,
    user: User = Depends(get_current_user)
):
    """Acknowledge an alert in both SQLite and the in-memory store."""
    # Update SQLite record if it exists
    db = SessionLocal()
    try:
        db_alert = db.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
        if db_alert:
            db_alert.acknowledged = True
            db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    # Also update in-memory deque for consistency during the same session
    for a in runtime.event_manager.alert_history:
        if a.get("alert_id") == alert_id:
            a["acknowledged"] = True
            break

    runtime.event_manager._broadcast({
        "type": "ALERT_ACKNOWLEDGED",
        "alert_id": alert_id,
        "acknowledged_by": user.email
    })
    return {"status": "acknowledged", "alert_id": alert_id}
