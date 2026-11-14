"""Database Models for Presence B2C Platform.
Implements household isolation, user roles, cameras, rules, events, and analytics per Section 18.
"""

from datetime import datetime, timezone
import uuid
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
)
from sqlalchemy.orm import relationship

from apps.api.app.core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Household(Base):
    __tablename__ = "households"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    name = Column(String(128), nullable=False)
    address = Column(String(256), nullable=True)
    timezone = Column(String(64), default="Asia/Kolkata")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    users = relationship("User", back_populates="household", cascade="all, delete-orphan")
    cameras = relationship("Camera", back_populates="household", cascade="all, delete-orphan")
    rules = relationship("CustomerRuleModel", back_populates="household", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    email = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    full_name = Column(String(128), nullable=False)
    role = Column(String(32), default="HOUSEHOLD_ADMIN")  # SUPER_ADMIN, HOUSEHOLD_ADMIN, MEMBER, VIEWER
    household_id = Column(String(64), ForeignKey("households.id"), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    household = relationship("Household", back_populates="users")


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), ForeignKey("households.id"), nullable=False)
    name = Column(String(128), nullable=False)
    rtsp_url = Column(String(512), nullable=True)
    stream_type = Column(String(32), default="synthetic")  # "rtsp", "synthetic", "file"
    location = Column(String(128), default="Living Room")
    is_online = Column(Boolean, default=True)
    resolution = Column(String(32), default="1920x1080")
    fps = Column(Integer, default=15)
    capabilities = Column(JSON, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    household = relationship("Household", back_populates="cameras")
    zones = relationship("MonitoringZoneModel", back_populates="camera", cascade="all, delete-orphan")


class MonitoringZoneModel(Base):
    __tablename__ = "monitoring_zones"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), ForeignKey("households.id"), nullable=False)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False)
    name = Column(String(128), nullable=False)
    polygon = Column(JSON, nullable=False)  # List of [x, y] coordinates
    zone_type = Column(String(64), default="restricted")  # "intrusion", "restricted", "cooking"
    loiter_threshold_seconds = Column(Integer, default=10)
    color = Column(String(16), default="#EF4444")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    camera = relationship("Camera", back_populates="zones")


class CustomerRuleModel(Base):
    __tablename__ = "customer_rules"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), ForeignKey("households.id"), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(String(256), nullable=True)
    camera_ids = Column(JSON, default=list)  # Empty means all cameras
    enabled = Column(Boolean, default=True)
    conditions = Column(JSON, nullable=False)
    schedule = Column(JSON, nullable=True)
    actions = Column(JSON, nullable=False)
    cooldown_seconds = Column(Integer, default=30)
    last_triggered_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    household = relationship("Household", back_populates="rules")


class AIEventRecord(Base):
    __tablename__ = "ai_events"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    module_id = Column(String(64), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    confidence = Column(Float, nullable=True)
    severity = Column(String(16), default="INFO", index=True)
    evidence_type = Column(String(32), default="video_bbox")
    zone_id = Column(String(64), nullable=True)
    metadata_json = Column(JSON, default=dict)
    snapshot_reference = Column(String(256), nullable=True)
    monitoring_uncertainty = Column(Float, default=0.0)
    acknowledged = Column(Boolean, default=False)


class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), nullable=False, index=True)
    rule_id = Column(String(64), nullable=True)
    rule_name = Column(String(128), nullable=False)
    event_type = Column(String(64), nullable=False)
    message = Column(Text, nullable=False)
    priority = Column(String(16), default="NORMAL", index=True)
    camera_id = Column(String(64), nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    recipient_group = Column(String(64), default="HOUSEHOLD_ADMINS")
    metadata_json = Column(JSON, default=dict)
    acknowledged = Column(Boolean, default=False)


class FaceEnrollment(Base):
    __tablename__ = "face_enrollments"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), nullable=False, index=True)
    name = Column(String(128), nullable=False)
    relation = Column(String(64), default="resident")  # child, parent, visitor
    consent_given = Column(Boolean, default=True)
    enrolled_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class VehicleLog(Base):
    __tablename__ = "vehicle_logs"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    plate_number = Column(String(32), nullable=False, index=True)
    vehicle_type = Column(String(32), default="car")
    is_registered = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class EnergyEstimateRecord(Base):
    __tablename__ = "energy_estimates"

    id = Column(String(64), primary_key=True, default=generate_uuid)
    household_id = Column(String(64), nullable=False, index=True)
    camera_id = Column(String(64), nullable=False)
    appliance_id = Column(String(64), nullable=False)
    appliance_type = Column(String(32), nullable=False)
    absence_minutes = Column(Float, default=0.0)
    waste_kwh = Column(Float, default=0.0)
    estimated_cost_inr = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
