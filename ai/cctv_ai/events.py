"""Event Normalizer and Dynamic Rule Engine Dispatcher for Presence B2C.
Validates, normalizes, deduplicates, and evaluates customer rules on all incoming AI events.
"""

from collections import deque
from datetime import datetime, timezone
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional
import uuid

from packages.event_schemas.events import StandardAIEvent, EventSeverity
from packages.rule_schemas.rules import ActionType, CustomerRule

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _requests = None  # type: ignore
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


class NotificationDelivery:
    """Represents a generated notification dispatched to user/household."""
    def __init__(
        self,
        alert_id: str,
        household_id: str,
        rule_id: Optional[str],
        rule_name: str,
        event_type: str,
        message: str,
        priority: str,
        camera_id: str,
        timestamp: datetime,
        recipient_group: str = "HOUSEHOLD_ADMINS",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.alert_id = alert_id
        self.household_id = household_id
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.event_type = event_type
        self.message = message
        self.priority = priority
        self.camera_id = camera_id
        self.timestamp = timestamp
        self.recipient_group = recipient_group
        self.metadata = metadata or {}
        self.acknowledged = False

    def to_dict(self) -> dict:
        return {
            "alert_id": self.alert_id,
            "household_id": self.household_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "event_type": self.event_type,
            "message": self.message,
            "priority": self.priority,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "recipient_group": self.recipient_group,
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


class EventManager:
    """Central event normalizer, deduplicator, and rule engine dispatcher."""

    def __init__(self):
        self.rules: Dict[str, CustomerRule] = {}
        self.event_history: deque = deque(maxlen=1000)
        self.alert_history: deque = deque(maxlen=500)
        self.subscribers: List[Callable[[dict], None]] = []
        self.rule_cooldowns: Dict[str, datetime] = {}  # rule_id -> last_triggered_dt

    def register_rule(self, rule: CustomerRule) -> None:
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str) -> Optional[CustomerRule]:
        return self.rules.pop(rule_id, None)

    def get_rules(self, household_id: Optional[str] = None) -> List[CustomerRule]:
        if household_id:
            return [r for r in self.rules.values() if r.household_id == household_id]
        return list(self.rules.values())

    def subscribe(self, callback: Callable[[dict], None]) -> None:
        self.subscribers.append(callback)

    def process_event(self, event: StandardAIEvent, ignore_schedule: bool = False) -> List[NotificationDelivery]:
        """Normalizes event, logs history, evaluates matching customer rules,
        and dispatches notifications.
        """
        now = datetime.now(timezone.utc)
        event_dict = event.to_dict()
        self.event_history.append(event_dict)

        # Notify raw subscribers (for real-time dashboard websockets)
        raw_msg = {"type": "AI_EVENT", "data": event_dict}
        self._broadcast(raw_msg)

        dispatched_alerts: List[NotificationDelivery] = []

        # Always notify for CRITICAL events even if no customer rule exists
        if event.severity == EventSeverity.CRITICAL:
            crit_alert = NotificationDelivery(
                alert_id=f"alert_{uuid.uuid4().hex[:10]}",
                household_id=event.household_id,
                rule_id=None,
                rule_name="CRITICAL SAFETY SYSTEM ALERT",
                event_type=event.event_type,
                message=f"CRITICAL SAFETY WARNING: {event.event_type} detected on camera '{event.camera_id}'!",
                priority="CRITICAL",
                camera_id=event.camera_id,
                timestamp=now,
                recipient_group="HOUSEHOLD_ADMINS",
                metadata=event.metadata
            )
            self.alert_history.append(crit_alert.to_dict())
            dispatched_alerts.append(crit_alert)
            self._broadcast({"type": "ALERT", "data": crit_alert.to_dict()})

        # Evaluate Customer Dynamic Rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            # Match household or default
            if rule.household_id and event.household_id:
                if (rule.household_id != "household_default" and 
                    event.household_id != "household_default" and 
                    rule.household_id != event.household_id):
                    continue

            # Check rule match
            if rule.matches(event_dict, check_time=now, ignore_schedule=ignore_schedule):
                # Check cooldown
                last_trig = self.rule_cooldowns.get(rule.id)
                if last_trig and (now - last_trig).total_seconds() < rule.cooldown_seconds:
                    continue

                self.rule_cooldowns[rule.id] = now
                rule.last_triggered_at = now

                # Dispatch actions
                for action in rule.actions:
                    # Construct message
                    default_msg = f"Rule '{rule.name}' triggered by {event.event_type} on {event.camera_id}."
                    context = {
                        "event": {
                            "type": event.event_type,
                            "camera_id": event.camera_id,
                            "zone": event.zone_id or "General",
                            "severity": event.severity.value if hasattr(event.severity, 'value') else event.severity,
                            "time": now.strftime("%I:%M %p"),
                        },
                        "metadata": event.metadata
                    }
                    rendered_msg = action.format_message(context, default_msg)

                    alert = NotificationDelivery(
                        alert_id=f"alert_{uuid.uuid4().hex[:10]}",
                        household_id=event.household_id,
                        rule_id=rule.id,
                        rule_name=rule.name,
                        event_type=event.event_type,
                        message=rendered_msg,
                        priority=action.priority,
                        camera_id=event.camera_id,
                        timestamp=now,
                        recipient_group=action.recipient_group,
                        metadata=event.metadata
                    )

                    self.alert_history.append(alert.to_dict())
                    dispatched_alerts.append(alert)
                    self._broadcast({"type": "ALERT", "data": alert.to_dict()})

                    # --- Side-effect delivery based on action type ---
                    if action.type == ActionType.WEBHOOK or action.type == "WEBHOOK":
                        webhook_url = getattr(action, "webhook_url", None)
                        if webhook_url and _REQUESTS_AVAILABLE:
                            payload = alert.to_dict()
                            def _post_webhook(url: str, data: dict) -> None:
                                try:
                                    _requests.post(url, json=data, timeout=10)
                                    logger.info(f"[WEBHOOK_DELIVERY] POST {url} -> status delivered")
                                except Exception as exc:
                                    logger.warning(f"[WEBHOOK_DELIVERY] POST {url} failed: {exc}")
                            threading.Thread(
                                target=_post_webhook,
                                args=(webhook_url, payload),
                                daemon=True
                            ).start()
                        elif not webhook_url:
                            logger.warning(
                                f"[WEBHOOK_DELIVERY] Rule '{rule.name}' action has no webhook_url configured — skipping."
                            )
                        else:
                            logger.warning(
                                "[WEBHOOK_DELIVERY] 'requests' library not available — skipping webhook dispatch."
                            )

                    elif action.type == ActionType.EMAIL or action.type == "EMAIL":
                        logger.info(
                            f"[EMAIL_DELIVERY] To: {action.recipient_group} "
                            f"Subject: Rule Alert — {rule.name} "
                            f"Msg: {rendered_msg}"
                        )

        return dispatched_alerts

    def _broadcast(self, msg: dict) -> None:
        for sub in self.subscribers:
            try:
                sub(msg)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
