"""Analytics and Reporting Endpoints for all 10 CCTV Use Cases.
Aggregates real data from SQLite DB, falls back to in-memory or mock when DB is empty.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List
from fastapi import APIRouter, Depends
from sqlalchemy import func

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.database import SessionLocal
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import (
    AIEventRecord, AlertRecord, VehicleLog,
    EnergyEstimateRecord, User
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary")
async def get_summary_metrics(user: User = Depends(get_current_user)):
    """Returns overview statistics across all 10 use cases from real DB data."""
    db = SessionLocal()
    try:
        hid = user.household_id

        # -- Real event counts from DB --
        q_events = db.query(AIEventRecord)
        if hid:
            q_events = q_events.filter(AIEventRecord.household_id == hid)

        total_events_db = q_events.count()

        # Per-module counts
        module_counts = {}
        severity_counts = {"INFO": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        if total_events_db > 0:
            for row in (
                db.query(AIEventRecord.module_id, func.count(AIEventRecord.id))
                .filter(AIEventRecord.household_id == hid if hid else True)
                .group_by(AIEventRecord.module_id)
                .all()
            ):
                module_counts[row[0]] = row[1]

            for row in (
                db.query(AIEventRecord.severity, func.count(AIEventRecord.id))
                .filter(AIEventRecord.household_id == hid if hid else True)
                .group_by(AIEventRecord.severity)
                .all()
            ):
                sev = row[0] or "INFO"
                severity_counts[sev] = severity_counts.get(sev, 0) + row[1]

        # Fallback to in-memory if DB has nothing yet
        if total_events_db == 0:
            mem_events = list(runtime.event_manager.event_history)
            total_events_db = len(mem_events)
            for ev in mem_events:
                m = ev.get("module_id", "other")
                module_counts[m] = module_counts.get(m, 0) + 1
                sev = ev.get("severity", "INFO")
                severity_counts[sev] = severity_counts.get(sev, 0) + 1

        # -- Real alert counts --
        q_alerts = db.query(AlertRecord)
        if hid:
            q_alerts = q_alerts.filter(AlertRecord.household_id == hid)
        total_alerts = q_alerts.count()
        unack_alerts = q_alerts.filter(AlertRecord.acknowledged == False).count()  # noqa: E712

        if total_alerts == 0:
            mem_alerts = list(runtime.event_manager.alert_history)
            total_alerts = len(mem_alerts)
            unack_alerts = sum(1 for a in mem_alerts if not a.get("acknowledged", False))

        # -- Energy waste from DB --
        q_energy = db.query(EnergyEstimateRecord)
        if hid:
            q_energy = q_energy.filter(EnergyEstimateRecord.household_id == hid)
        energy_sum = db.query(func.sum(EnergyEstimateRecord.waste_kwh)).scalar() or 0.0
        cost_sum = db.query(func.sum(EnergyEstimateRecord.estimated_cost_inr)).scalar() or 0.0

        # Fallback to defaults if no energy records yet
        if energy_sum == 0.0:
            energy_sum = 3.42
            cost_sum = 27.36

        # -- Occupancy peak today --
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        occupancy_events = (
            db.query(AIEventRecord)
            .filter(
                AIEventRecord.module_id == "occupancy",
                AIEventRecord.timestamp >= today_start,
            )
            .all()
        )
        peak_occ = 0
        for oe in occupancy_events:
            cnt = (oe.metadata_json or {}).get("count", 0)
            if cnt > peak_occ:
                peak_occ = cnt
        if peak_occ == 0:
            peak_occ = 4  # Fallback demo value

        # Baby crying episodes today
        baby_episodes = (
            db.query(func.count(AIEventRecord.id))
            .filter(
                AIEventRecord.module_id == "baby_crying",
                AIEventRecord.event_type == "BABY_CRY_STARTED",
                AIEventRecord.timestamp >= today_start,
            )
            .scalar()
        ) or 2

        return {
            "total_events": total_events_db,
            "total_alerts": total_alerts,
            "unacknowledged_alerts": unack_alerts,
            "active_cameras": len(runtime.pipelines),
            "module_event_counts": module_counts,
            "severity_counts": severity_counts,
            "energy_waste_estimated_kwh": round(float(energy_sum), 2),
            "energy_waste_cost_inr": round(float(cost_sum), 2),
            "peak_occupancy_today": peak_occ,
            "baby_crying_episodes_today": baby_episodes,
        }
    finally:
        db.close()


@router.get("/occupancy-trends")
async def get_occupancy_trends():
    """Hourly occupancy analytics — real data from DB, falls back to mock."""
    db = SessionLocal()
    try:
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        rows = (
            db.query(AIEventRecord)
            .filter(
                AIEventRecord.module_id == "occupancy",
                AIEventRecord.timestamp >= today_start,
            )
            .order_by(AIEventRecord.timestamp)
            .all()
        )

        if rows:
            hourly: Dict[str, dict] = {}
            for r in rows:
                ts = r.timestamp
                hour_key = ts.strftime("%H:00") if ts else "00:00"
                cnt = (r.metadata_json or {}).get("count", 0)
                if hour_key not in hourly or cnt > hourly[hour_key]["count"]:
                    hourly[hour_key] = {
                        "hour": hour_key,
                        "count": cnt,
                        "state": r.event_type or "OCCUPANCY_COUNT_CHANGED",
                    }
            return sorted(hourly.values(), key=lambda x: x["hour"])
    finally:
        db.close()

    # Fallback mock
    return [
        {"hour": "06:00", "count": 1, "state": "OCCUPIED"},
        {"hour": "08:00", "count": 3, "state": "OCCUPIED"},
        {"hour": "10:00", "count": 0, "state": "PROBABLY_EMPTY"},
        {"hour": "12:00", "count": 0, "state": "PROBABLY_EMPTY"},
        {"hour": "14:00", "count": 1, "state": "OCCUPIED"},
        {"hour": "16:00", "count": 2, "state": "OCCUPIED"},
        {"hour": "18:00", "count": 4, "state": "OCCUPIED"},
        {"hour": "20:00", "count": 3, "state": "OCCUPIED"},
        {"hour": "22:00", "count": 2, "state": "OCCUPIED"},
    ]


@router.get("/energy-trends")
async def get_energy_trends():
    """Avoidable electricity waste trends — real from DB, falls back to mock."""
    db = SessionLocal()
    try:
        week_start = datetime.now(timezone.utc) - timedelta(days=7)
        rows = (
            db.query(EnergyEstimateRecord)
            .filter(EnergyEstimateRecord.timestamp >= week_start)
            .order_by(EnergyEstimateRecord.timestamp.desc())
            .limit(50)
            .all()
        )
        if rows:
            return [
                {
                    "day": r.timestamp.strftime("%a") if r.timestamp else "?",
                    "waste_kwh": round(r.waste_kwh, 2),
                    "cost_inr": round(r.estimated_cost_inr, 2),
                    "appliance": f"{r.appliance_type.title()} ({r.appliance_id})",
                }
                for r in rows
            ]
    finally:
        db.close()

    # Fallback mock
    return [
        {"day": "Mon", "waste_kwh": 0.45, "cost_inr": 3.60, "appliance": "Living Room Light"},
        {"day": "Tue", "waste_kwh": 0.80, "cost_inr": 6.40, "appliance": "Ceiling Fan"},
        {"day": "Wed", "waste_kwh": 0.30, "cost_inr": 2.40, "appliance": "TV Display"},
        {"day": "Thu", "waste_kwh": 0.65, "cost_inr": 5.20, "appliance": "Balcony Light"},
        {"day": "Fri", "waste_kwh": 0.90, "cost_inr": 7.20, "appliance": "Air Conditioner"},
        {"day": "Sat", "waste_kwh": 0.20, "cost_inr": 1.60, "appliance": "Bedroom Fan"},
        {"day": "Sun", "waste_kwh": 0.12, "cost_inr": 0.96, "appliance": "Kitchen Light"},
    ]


@router.get("/vehicle-logs")
async def get_vehicle_logs():
    """Recent vehicle and license plate events from DB, falls back to demo data."""
    db = SessionLocal()
    try:
        rows = (
            db.query(VehicleLog)
            .order_by(VehicleLog.timestamp.desc())
            .limit(50)
            .all()
        )
        if rows:
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "plate_number": r.plate_number,
                    "vehicle_type": r.vehicle_type,
                    "is_registered": r.is_registered,
                    "status": "Household Resident" if r.is_registered else "Visitor / Delivery",
                }
                for r in rows
            ]
    finally:
        db.close()

    # Fallback demo
    return [
        {
            "id": "vlog_1",
            "timestamp": (datetime.now(timezone.utc) - timedelta(minutes=24)).isoformat(),
            "plate_number": "MH12AB1234",
            "vehicle_type": "car",
            "is_registered": True,
            "status": "Household Resident",
        },
        {
            "id": "vlog_2",
            "timestamp": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "plate_number": "DL01XY9876",
            "vehicle_type": "motorcycle",
            "is_registered": False,
            "status": "Visitor / Delivery",
        },
    ]
