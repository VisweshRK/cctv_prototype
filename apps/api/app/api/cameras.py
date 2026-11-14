"""Camera Management and Compatibility Diagnostics Endpoints.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.app.api.deps import get_current_user
from apps.api.app.cameras.compatibility import CameraCompatibilityChecker
from apps.api.app.core.database import get_db
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import Camera, User

router = APIRouter(prefix="/cameras", tags=["Cameras"])


class CameraCreateRequest(BaseModel):
    name: str
    rtsp_url: Optional[str] = None
    stream_type: str = "synthetic"  # "synthetic", "rtsp", "file"
    location: str = "Living Room"


class CameraProbeRequest(BaseModel):
    ip: str
    rtsp_port: int = 554
    onvif_port: int = 80
    username: Optional[str] = None
    password: Optional[str] = None
    test_stream_url: Optional[str] = None


@router.get("")
def list_cameras(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = select(Camera).where(Camera.household_id == user.household_id)
    cameras = db.execute(stmt).scalars().all()

    if not cameras:
        default_cam = Camera(
            id="camera_main",
            household_id=user.household_id or "household_default",
            name="Main Living Area Camera",
            stream_type="synthetic",
            location="Living Room",
            is_online=True,
            capabilities={
                "audio": True,
                "ptz": False,
                "night_vision": True,
                "fps": 15,
                "resolution": "1920x1080"
            }
        )
        db.add(default_cam)
        db.commit()
        db.refresh(default_cam)
        cameras = [default_cam]

    results = []
    for c in cameras:
        pipeline = runtime.pipelines.get(c.id)
        fps = pipeline.metrics["inference_fps"] if pipeline else 15.0
        active_objs = pipeline.metrics["active_objects_count"] if pipeline else 0
        results.append({
            "id": c.id,
            "household_id": c.household_id,
            "name": c.name,
            "rtsp_url": c.rtsp_url,
            "stream_type": c.stream_type,
            "location": c.location,
            "is_online": c.is_online,
            "fps": fps,
            "active_objects": active_objs,
            "capabilities": c.capabilities,
        })
    return results


@router.post("")
def add_camera(
    req: CameraCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    cam = Camera(
        household_id=user.household_id or "household_default",
        name=req.name,
        rtsp_url=req.rtsp_url,
        stream_type=req.stream_type,
        location=req.location,
        is_online=True,
        capabilities={"audio": True, "night_vision": True, "fps": 15}
    )
    db.add(cam)
    db.commit()
    db.refresh(cam)

    runtime.get_or_create_pipeline(cam.id, cam.stream_type)
    return cam


@router.post("/probe")
def probe_camera_compatibility(req: CameraProbeRequest):
    """Executes non-destructive camera hardware compatibility check."""
    checker = CameraCompatibilityChecker()
    report = checker.probe(
        ip=req.ip,
        rtsp_port=req.rtsp_port,
        onvif_port=req.onvif_port,
        username=req.username,
        password=req.password,
        test_stream_url=req.test_stream_url
    )
    return report.to_dict()
