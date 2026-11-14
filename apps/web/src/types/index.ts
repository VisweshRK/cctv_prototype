/**
 * TypeScript Type Definitions for Presence B2C CCTV AI Platform.
 */

export type UserRole = "SUPER_ADMIN" | "HOUSEHOLD_ADMIN" | "MEMBER" | "VIEWER";

export interface User {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  household_id: string;
}

export interface Camera {
  id: string;
  name: string;
  stream_type: "synthetic" | "rtsp" | "file";
  location: string;
  is_online: boolean;
  fps: number;
  active_objects: number;
  capabilities: {
    audio?: boolean;
    ptz?: boolean;
    night_vision?: boolean;
    fps?: number;
    resolution?: string;
  };
}

export interface MonitoringZone {
  zone_id: string;
  name: string;
  camera_id: string;
  polygon: [number, number][];
  zone_type: "intrusion" | "restricted" | "cooking";
  loiter_threshold_seconds: number;
  color: string;
}

export interface RuleConditionItem {
  field: string;
  operator: "EQ" | "NEQ" | "GT" | "GTE" | "LT" | "LTE" | "CONTAINS" | "IN" | "STARTS_WITH";
  value: any;
}

export interface RuleConditionGroup {
  operator: "AND" | "OR";
  items: (RuleConditionItem | RuleConditionGroup)[];
}

export interface RuleAction {
  type: "PUSH_NOTIFICATION" | "IN_APP_ALERT" | "WEBHOOK" | "EMAIL";
  recipient_group: string;
  custom_message_template?: string;
  priority: "LOW" | "NORMAL" | "HIGH" | "CRITICAL";
}

export interface CustomerRule {
  id: string;
  name: string;
  description?: string;
  camera_ids: string[];
  enabled: boolean;
  conditions: RuleConditionGroup;
  schedule?: {
    timezone: string;
    start_time?: string;
    end_time?: string;
    days_of_week?: number[];
  };
  actions: RuleAction[];
  cooldown_seconds: number;
}

export interface AlertNotification {
  alert_id: string;
  household_id: string;
  rule_id?: string;
  rule_name: string;
  event_type: string;
  message: string;
  priority: "LOW" | "NORMAL" | "HIGH" | "CRITICAL";
  camera_id: string;
  timestamp: string;
  recipient_group: string;
  metadata?: Record<string, any>;
  acknowledged: boolean;
}

export interface AIEvent {
  event_id: string;
  household_id: string;
  camera_id: string;
  module_id: string;
  event_type: string;
  timestamp: string;
  confidence?: number;
  severity: "INFO" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  evidence_type: string;
  zone_id?: string;
  metadata?: Record<string, any>;
  monitoring_uncertainty: number;
  acknowledged: boolean;
}

export interface ModelMetadata {
  model_name: string;
  module_id: string;
  source_repository: string;
  exact_checkpoint_source: string;
  license: string;
  commercial_use_permission: boolean;
  dataset_source: string;
  dataset_rights: string;
  target_runtime: string;
  hardware_compatibility: string;
  validation_status: "VALIDATED" | "EXPERIMENTAL" | "NOT_READY" | "FALLBACK_CV";
  model_version: string;
  notes: string;
}

export interface CompatibilityReport {
  camera_ip: string;
  port: number;
  camera_model: string;
  ip_reachable: boolean;
  rtsp_available: boolean;
  onvif_available: boolean;
  video_codec?: string;
  audio_codec?: string;
  microphone_detected: boolean;
  resolution?: string;
  fps?: number;
  ptz_supported: boolean;
  deployment_mode: "MODE_A_EXTERNAL_AI" | "MODE_B_ONBOARD_AI" | "MODE_C_UNSUPPORTED";
  tested_at: string;
  limitations: string[];
}
