"""Monitoring Zones API Endpoints: Polygon definition and loitering settings.
"""

from typing import List, Tuple
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ai.cctv_ai.zones import MonitoringZone
from apps.api.app.api.deps import get_current_user
from apps.api.app.core.database import get_db
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import MonitoringZoneModel, User

router = APIRouter(prefix="/zones", tags=["Zones"])


class ZoneCreateRequest(BaseModel):
    name: str
    camera_id: str
    polygon: List[List[int]]  # [[x1, y1], [x2, y2], ...]
    zone_type: str = "restricted"  # "intrusion", "restricted", "cooking"
    loiter_threshold_seconds: int = 10
    color: str = "#EF4444"


@router.get("")
async def list_zones(
    camera_id: str = "camera_main",
    user: User = Depends(get_current_user)
):
    zones = runtime.zone_manager.get_camera_zones(camera_id)
    if not zones:
        # Seed default sample entrance and balcony zones
        z1 = MonitoringZone(
            zone_id="zone_entrance",
            name="Front Door Entrance",
            camera_id=camera_id,
            polygon=[(50, 150), (170, 150), (170, 350), (50, 350)],
            zone_type="intrusion",
            loiter_threshold_seconds=5,
            color="#3B82F6"
        )
        z2 = MonitoringZone(
            zone_id="zone_balcony",
            name="Balcony Restricted Ledge",
            camera_id=camera_id,
            polygon=[(400, 200), (600, 200), (600, 420), (400, 420)],
            zone_type="restricted",
            loiter_threshold_seconds=8,
            color="#EF4444"
        )
        runtime.zone_manager.add_zone(z1)
        runtime.zone_manager.add_zone(z2)
        zones = [z1, z2]

    return [z.to_dict() for z in zones]


@router.post("")
async def create_zone(
    req: ZoneCreateRequest,
    user: User = Depends(get_current_user)
):
    import uuid
    zid = f"zone_{uuid.uuid4().hex[:8]}"
    polygon = [tuple(p) for p in req.polygon]
    z = MonitoringZone(
        zone_id=zid,
        name=req.name,
        camera_id=req.camera_id,
        polygon=polygon,
        zone_type=req.zone_type,
        loiter_threshold_seconds=req.loiter_threshold_seconds,
        color=req.color
    )
    runtime.zone_manager.add_zone(z)
    return z.to_dict()


@router.delete("/{zone_id}")
async def delete_zone(
    zone_id: str,
    user: User = Depends(get_current_user)
):
    deleted = runtime.zone_manager.remove_zone(zone_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Zone not found")
    return {"status": "deleted", "zone_id": zone_id}
