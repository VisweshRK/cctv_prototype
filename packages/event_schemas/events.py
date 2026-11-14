"""Event Schemas for Presence B2C CCTV AI Platform.
Standardized event contract matching Section 4 requirements.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class EventSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StandardEventType(str, Enum):
    # Intrusion
    INTRUSION_CANDIDATE = "INTRUSION_CANDIDATE"
    PROTECTED_ZONE_ENTERED = "PROTECTED_ZONE_ENTERED"
    PROTECTED_ZONE_EXITED = "PROTECTED_ZONE_EXITED"
    MONITORING_UNAVAILABLE = "MONITORING_UNAVAILABLE"

    # Tampering
    CAMERA_COVERED = "CAMERA_COVERED"
    CAMERA_BLURRED = "CAMERA_BLURRED"
    CAMERA_MOVED = "CAMERA_MOVED"
    CAMERA_DISCONNECTED = "CAMERA_DISCONNECTED"
    CAMERA_STREAM_FROZEN = "CAMERA_STREAM_FROZEN"
    CAMERA_RESTORED = "CAMERA_RESTORED"

    # Face Recognition
    ENROLLED_PERSON_RECOGNIZED = "ENROLLED_PERSON_RECOGNIZED"
    PERSON_NOT_RECOGNIZED = "PERSON_NOT_RECOGNIZED"
    FACE_RECOGNITION_UNCERTAIN = "FACE_RECOGNITION_UNCERTAIN"

    # Restricted Area
    RESTRICTED_ZONE_ENTERED = "RESTRICTED_ZONE_ENTERED"
    RESTRICTED_ZONE_LOITERING = "RESTRICTED_ZONE_LOITERING"
    RESTRICTED_ZONE_EXITED = "RESTRICTED_ZONE_EXITED"

    # Fall
    POSSIBLE_FALL = "POSSIBLE_FALL"
    PERSON_DOWN = "PERSON_DOWN"
    PERSON_VISIBILITY_LOST = "PERSON_VISIBILITY_LOST"
    FALL_MONITORING_UNAVAILABLE = "FALL_MONITORING_UNAVAILABLE"

    # Occupancy
    OCCUPIED = "OCCUPIED"
    PROBABLY_EMPTY = "PROBABLY_EMPTY"
    OCCUPANCY_UNKNOWN = "OCCUPANCY_UNKNOWN"
    OCCUPANCY_COUNT_CHANGED = "OCCUPANCY_COUNT_CHANGED"

    # Energy / Power When Absent
    ROOM_PROBABLY_EMPTY = "ROOM_PROBABLY_EMPTY"
    LIGHT_POSSIBLY_LEFT_ON = "LIGHT_POSSIBLY_LEFT_ON"
    FAN_POSSIBLY_LEFT_ON = "FAN_POSSIBLY_LEFT_ON"
    TV_POSSIBLY_LEFT_ON = "TV_POSSIBLY_LEFT_ON"
    POTENTIAL_ENERGY_WASTE = "POTENTIAL_ENERGY_WASTE"
    ENERGY_ESTIMATE_UNCERTAIN = "ENERGY_ESTIMATE_UNCERTAIN"
    METER_UNAVAILABLE = "METER_UNAVAILABLE"

    # Fire and Smoke
    FIRE_CANDIDATE = "FIRE_CANDIDATE"
    SMOKE_CANDIDATE = "SMOKE_CANDIDATE"
    POSSIBLE_DANGEROUS_FIRE = "POSSIBLE_DANGEROUS_FIRE"
    EXPECTED_COOKING_FLAME = "EXPECTED_COOKING_FLAME"
    VISIBILITY_DEGRADED = "VISIBILITY_DEGRADED"
    CONNECTED_FIRE_ALARM = "CONNECTED_FIRE_ALARM"

    # Vehicle and LPR
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    PLATE_READ = "PLATE_READ"
    PLATE_UNREADABLE = "PLATE_UNREADABLE"
    REGISTERED_VEHICLE = "REGISTERED_VEHICLE"
    UNRECOGNIZED_PLATE = "UNRECOGNIZED_PLATE"
    VEHICLE_ENTERED_ZONE = "VEHICLE_ENTERED_ZONE"

    # Baby Crying
    BABY_CRY_STARTED = "BABY_CRY_STARTED"
    BABY_CRY_CONTINUES = "BABY_CRY_CONTINUES"
    BABY_CRY_ENDED = "BABY_CRY_ENDED"
    BABY_CRY_WITH_RAISED_VOICES = "BABY_CRY_WITH_RAISED_VOICES"
    BABY_CRY_MONITORING_UNAVAILABLE = "BABY_CRY_MONITORING_UNAVAILABLE"


@dataclass
class StandardAIEvent:
    """Standardized AI Event emitted by all 10 modules."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    household_id: str = "household_default"
    camera_id: str = "camera_main"
    module_id: str = ""
    event_type: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    confidence: Optional[float] = None
    evidence_type: str = "video_bbox"  # "video_bbox", "audio_waveform", "scene_metric", "pose_keypoints", "ocr_text"
    model_version: str = "1.0.0"
    zone_id: Optional[str] = None
    severity: EventSeverity = EventSeverity.INFO
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshot_reference: Optional[str] = None
    monitoring_uncertainty: float = 0.0  # 0.0 = certain, 1.0 = completely uncertain
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "household_id": self.household_id,
            "camera_id": self.camera_id,
            "module_id": self.module_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "confidence": self.confidence,
            "evidence_type": self.evidence_type,
            "model_version": self.model_version,
            "zone_id": self.zone_id,
            "severity": self.severity.value if isinstance(self.severity, EventSeverity) else self.severity,
            "metadata": self.metadata,
            "snapshot_reference": self.snapshot_reference,
            "monitoring_uncertainty": self.monitoring_uncertainty,
            "acknowledged": self.acknowledged,
        }
