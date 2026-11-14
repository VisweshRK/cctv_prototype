"""Rule Schemas and Safe Evaluator for Presence B2C Platform.
Implements typed allowlisted rule DSL per Section 15.
Customers can never execute arbitrary code.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Union
import uuid


class RuleOperator(str, Enum):
    EQ = "EQ"
    NEQ = "NEQ"
    GT = "GT"
    GTE = "GTE"
    LT = "LT"
    LTE = "LTE"
    CONTAINS = "CONTAINS"
    IN = "IN"
    NOT_IN = "NOT_IN"
    STARTS_WITH = "STARTS_WITH"


class LogicalOperator(str, Enum):
    AND = "AND"
    OR = "OR"


class ActionType(str, Enum):
    PUSH_NOTIFICATION = "PUSH_NOTIFICATION"
    IN_APP_ALERT = "IN_APP_ALERT"
    WEBHOOK = "WEBHOOK"
    EMAIL = "EMAIL"


# Allowed fields to query in rules to prevent injection
ALLOWLISTED_FIELDS = {
    "event.type",
    "event.severity",
    "event.camera_id",
    "event.zone_id",
    "event.confidence",
    "event.monitoring_uncertainty",
    "event.module_id",
    "metadata.person_name",
    "metadata.is_known",
    "metadata.count",
    "metadata.plate_number",
    "metadata.appliance_type",
    "metadata.duration_seconds",
    "metadata.fall_type",
    "metadata.is_fire",
    "metadata.is_smoke",
}


@dataclass
class RuleConditionItem:
    field: str
    operator: RuleOperator
    value: Any

    def evaluate(self, event_context: Dict[str, Any]) -> bool:
        if self.field not in ALLOWLISTED_FIELDS:
            return False

        # Extract value from nested context
        parts = self.field.split(".")
        current = event_context
        for p in parts:
            if isinstance(current, dict) and p in current:
                current = current[p]
            else:
                current = None
                break

        actual_val = current
        target_val = self.value

        if self.operator == RuleOperator.EQ:
            return str(actual_val).lower() == str(target_val).lower()
        elif self.operator == RuleOperator.NEQ:
            return str(actual_val).lower() != str(target_val).lower()
        elif self.operator == RuleOperator.CONTAINS:
            return target_val is not None and str(target_val).lower() in str(actual_val).lower()
        elif self.operator == RuleOperator.STARTS_WITH:
            return target_val is not None and str(actual_val).lower().startswith(str(target_val).lower())
        elif self.operator == RuleOperator.IN:
            if isinstance(target_val, list):
                return str(actual_val) in [str(x) for x in target_val]
            return False
        elif self.operator == RuleOperator.NOT_IN:
            if isinstance(target_val, list):
                return str(actual_val) not in [str(x) for x in target_val]
            return True
        elif self.operator in (RuleOperator.GT, RuleOperator.GTE, RuleOperator.LT, RuleOperator.LTE):
            try:
                a_num = float(actual_val) if actual_val is not None else 0.0
                t_num = float(target_val) if target_val is not None else 0.0
                if self.operator == RuleOperator.GT:
                    return a_num > t_num
                elif self.operator == RuleOperator.GTE:
                    return a_num >= t_num
                elif self.operator == RuleOperator.LT:
                    return a_num < t_num
                elif self.operator == RuleOperator.LTE:
                    return a_num <= t_num
            except (ValueError, TypeError):
                return False

        return False


@dataclass
class RuleConditionGroup:
    operator: LogicalOperator = LogicalOperator.AND
    items: List[Union[RuleConditionItem, "RuleConditionGroup"]] = field(default_factory=list)

    def evaluate(self, event_context: Dict[str, Any]) -> bool:
        if not self.items:
            return True

        if self.operator == LogicalOperator.AND:
            for item in self.items:
                if not item.evaluate(event_context):
                    return False
            return True
        elif self.operator == LogicalOperator.OR:
            for item in self.items:
                if item.evaluate(event_context):
                    return True
            return False
        return False


@dataclass
class RuleSchedule:
    timezone: str = "Asia/Kolkata"
    start_time: Optional[str] = None  # "HH:MM" e.g. "14:00"
    end_time: Optional[str] = None    # "HH:MM" e.g. "17:30"
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])  # 0=Mon, 6=Sun

    def is_active_now(self, dt: Optional[datetime] = None) -> bool:
        if not dt:
            dt = datetime.now(timezone.utc)
        # Check day of week
        if dt.weekday() not in self.days_of_week:
            return False

        if not self.start_time or not self.end_time:
            return True

        try:
            s_hour, s_min = map(int, self.start_time.split(":"))
            e_hour, e_min = map(int, self.end_time.split(":"))
            start = time(s_hour, s_min)
            end = time(e_hour, e_min)
            current_time = dt.time()

            if start <= end:
                return start <= current_time <= end
            else:
                # Overnight schedule e.g. 23:00 to 06:00
                return current_time >= start or current_time <= end
        except Exception:
            return True


@dataclass
class RuleAction:
    type: ActionType
    recipient_group: str = "HOUSEHOLD_ADMINS"  # "HOUSEHOLD_ADMINS", "ALL_MEMBERS", "CAREGIVERS", "POLICE_CONTACT"
    custom_message_template: Optional[str] = None  # e.g. "Child {metadata.person_name} came from school at {event.time}!"
    priority: str = "NORMAL"  # "LOW", "NORMAL", "HIGH", "CRITICAL"
    webhook_url: Optional[str] = None  # Target URL for WEBHOOK action type

    def format_message(self, event_context: Dict[str, Any], default_text: str = "") -> str:
        if not self.custom_message_template:
            return default_text

        msg = self.custom_message_template
        # Replace occurrences like {metadata.person_name}
        pattern = r"\{([a-zA-Z0-9_\.]+)\}"
        def repl(match):
            key = match.group(1)
            parts = key.split(".")
            cur = event_context
            for p in parts:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    return match.group(0)
            return str(cur)

        return re.sub(pattern, repl, msg)


@dataclass
class CustomerRule:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    household_id: str = "household_default"
    name: str = "Untitled Rule"
    description: str = ""
    camera_ids: List[str] = field(default_factory=list)  # Empty means all cameras in household
    enabled: bool = True
    conditions: RuleConditionGroup = field(default_factory=RuleConditionGroup)
    schedule: Optional[RuleSchedule] = None
    actions: List[RuleAction] = field(default_factory=list)
    cooldown_seconds: int = 30
    last_triggered_at: Optional[datetime] = None

    def matches(self, event: Dict[str, Any], check_time: Optional[datetime] = None, ignore_schedule: bool = False) -> bool:
        if not self.enabled:
            return False

        # Check camera match
        if self.camera_ids and event.get("camera_id") not in self.camera_ids:
            return False

        # Check schedule
        if not ignore_schedule and self.schedule and not self.schedule.is_active_now(check_time):
            return False

        # Flatten context for condition evaluator
        context = {
            "event": {
                "type": event.get("event_type"),
                "severity": event.get("severity"),
                "camera_id": event.get("camera_id"),
                "zone_id": event.get("zone_id"),
                "confidence": event.get("confidence"),
                "monitoring_uncertainty": event.get("monitoring_uncertainty", 0.0),
                "module_id": event.get("module_id"),
                "time": event.get("timestamp"),
            },
            "metadata": event.get("metadata", {})
        }

        # Evaluate conditions
        return self.conditions.evaluate(context)
