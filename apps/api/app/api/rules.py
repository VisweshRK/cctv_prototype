"""Dynamic Customer Rules API Endpoints.
Allows customers to construct IF / AND / OR / THEN rules, simulate them, and customize notification templates.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.app.api.deps import get_current_user
from apps.api.app.core.database import get_db
from apps.api.app.core.runtime import runtime
from apps.api.app.models.entities import CustomerRuleModel, User
from packages.rule_schemas.rules import (
    CustomerRule,
    RuleConditionGroup,
    RuleConditionItem,
    RuleOperator,
    RuleAction,
    ActionType
)

router = APIRouter(prefix="/rules", tags=["Rules"])


class RuleCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    camera_ids: List[str] = []
    enabled: bool = True
    conditions: Dict[str, Any]
    schedule: Optional[Dict[str, Any]] = None
    actions: List[Dict[str, Any]]
    cooldown_seconds: int = 30


class RuleSimulateRequest(BaseModel):
    rule: Dict[str, Any]
    sample_event: Dict[str, Any]


def parse_condition_group(data: Dict[str, Any]) -> RuleConditionGroup:
    op = data.get("operator", "AND")
    raw_items = data.get("items", [])
    parsed_items = []
    for item in raw_items:
        if "items" in item:
            parsed_items.append(parse_condition_group(item))
        else:
            parsed_items.append(RuleConditionItem(
                field=item["field"],
                operator=RuleOperator(item["operator"]),
                value=item["value"]
            ))
    return RuleConditionGroup(operator=op, items=parsed_items)


@router.get("")
def list_rules(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = select(CustomerRuleModel).where(CustomerRuleModel.household_id == user.household_id)
    rules = db.execute(stmt).scalars().all()

    if not rules:
        seed_rule_data = [
            {
                "name": "Dangerous Fire & Smoke Warning",
                "description": "Critical siren if uncontrolled fire or smoke is detected outside cooking zones.",
                "camera_ids": ["camera_main"],
                "enabled": True,
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"field": "event.type", "operator": "EQ", "value": "POSSIBLE_DANGEROUS_FIRE"}
                    ]
                },
                "schedule": None,
                "actions": [
                    {
                        "type": "PUSH_NOTIFICATION",
                        "recipient_group": "HOUSEHOLD_ADMINS",
                        "custom_message_template": "CRITICAL EMERGENCY: Fire detected on {event.camera_id} at {event.time}!",
                        "priority": "CRITICAL"
                    }
                ],
                "cooldown_seconds": 30
            },
            {
                "name": "Electricity Waste During Absence Alert",
                "description": "Alerts when lights or appliances are left on while nobody is in the room.",
                "camera_ids": ["camera_main"],
                "enabled": True,
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"field": "event.type", "operator": "EQ", "value": "LIGHT_POSSIBLY_LEFT_ON"}
                    ]
                },
                "schedule": None,
                "actions": [
                    {
                        "type": "PUSH_NOTIFICATION",
                        "recipient_group": "HOUSEHOLD_ADMINS",
                        "custom_message_template": "Energy Alert: Lights left on while room is empty at {event.time}! Est. waste: {metadata.estimated_waste_kwh} kWh.",
                        "priority": "NORMAL"
                    }
                ],
                "cooldown_seconds": 60
            },
            {
                "name": "Baby Crying Monitor Alert",
                "description": "Notifies parents immediately when infant crying or distress sound is detected.",
                "camera_ids": ["camera_main"],
                "enabled": True,
                "conditions": {
                    "operator": "AND",
                    "items": [
                        {"field": "event.type", "operator": "EQ", "value": "BABY_CRY_STARTED"}
                    ]
                },
                "schedule": None,
                "actions": [
                    {
                        "type": "PUSH_NOTIFICATION",
                        "recipient_group": "HOUSEHOLD_ADMINS",
                        "custom_message_template": "Baby Monitor: Infant crying detected on {event.camera_id} at {event.time}!",
                        "priority": "HIGH"
                    }
                ],
                "cooldown_seconds": 60
            }
        ]

        seeded = []
        for r_dict in seed_rule_data:
            m = CustomerRuleModel(
                household_id=user.household_id or "household_default",
                name=r_dict["name"],
                description=r_dict["description"],
                camera_ids=r_dict["camera_ids"],
                enabled=r_dict["enabled"],
                conditions=r_dict["conditions"],
                schedule=r_dict["schedule"],
                actions=r_dict["actions"],
                cooldown_seconds=r_dict["cooldown_seconds"]
            )
            db.add(m)
            seeded.append(m)
        db.commit()
        rules = seeded

        for r in rules:
            parsed_cond = parse_condition_group(r.conditions)
            actions = [
                RuleAction(
                    type=ActionType(a["type"]),
                    recipient_group=a.get("recipient_group", "HOUSEHOLD_ADMINS"),
                    custom_message_template=a.get("custom_message_template"),
                    priority=a.get("priority", "NORMAL")
                )
                for a in r.actions
            ]
            c_rule = CustomerRule(
                id=r.id,
                household_id=r.household_id,
                name=r.name,
                description=r.description or "",
                camera_ids=r.camera_ids or [],
                enabled=r.enabled,
                conditions=parsed_cond,
                actions=actions,
                cooldown_seconds=r.cooldown_seconds
            )
            runtime.event_manager.register_rule(c_rule)

    return rules


def load_all_rules_into_runtime():
    """Loads all rules from the database into runtime.event_manager on server startup."""
    from apps.api.app.core.database import SessionLocal
    with SessionLocal() as db:
        stmt = select(CustomerRuleModel)
        rules = db.execute(stmt).scalars().all()
        for r in rules:
            try:
                parsed_cond = parse_condition_group(r.conditions)
                actions = [
                    RuleAction(
                        type=ActionType(a["type"]),
                        recipient_group=a.get("recipient_group", "HOUSEHOLD_ADMINS"),
                        custom_message_template=a.get("custom_message_template"),
                        priority=a.get("priority", "NORMAL")
                    )
                    for a in r.actions
                ]
                c_rule = CustomerRule(
                    id=r.id,
                    household_id=r.household_id,
                    name=r.name,
                    description=r.description or "",
                    camera_ids=r.camera_ids or [],
                    enabled=r.enabled,
                    conditions=parsed_cond,
                    actions=actions,
                    cooldown_seconds=r.cooldown_seconds
                )
                runtime.event_manager.register_rule(c_rule)
            except Exception as e:
                pass


@router.post("")
def create_rule(
    req: RuleCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    model = CustomerRuleModel(
        household_id=user.household_id or "household_default",
        name=req.name,
        description=req.description,
        camera_ids=req.camera_ids,
        enabled=req.enabled,
        conditions=req.conditions,
        schedule=req.schedule,
        actions=req.actions,
        cooldown_seconds=req.cooldown_seconds
    )
    db.add(model)
    db.commit()
    db.refresh(model)

    parsed_cond = parse_condition_group(model.conditions)
    actions = [
        RuleAction(
            type=ActionType(a["type"]),
            recipient_group=a.get("recipient_group", "HOUSEHOLD_ADMINS"),
            custom_message_template=a.get("custom_message_template"),
            priority=a.get("priority", "NORMAL")
        )
        for a in model.actions
    ]
    c_rule = CustomerRule(
        id=model.id,
        household_id=model.household_id,
        name=model.name,
        description=model.description or "",
        camera_ids=model.camera_ids or [],
        enabled=model.enabled,
        conditions=parsed_cond,
        actions=actions,
        cooldown_seconds=model.cooldown_seconds
    )
    runtime.event_manager.register_rule(c_rule)

    return model


@router.put("/{rule_id}")
def update_rule(
    rule_id: str,
    req: RuleCreateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = select(CustomerRuleModel).where(CustomerRuleModel.id == rule_id)
    model = db.execute(stmt).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Rule not found")

    model.name = req.name
    model.description = req.description
    model.camera_ids = req.camera_ids
    model.enabled = req.enabled
    model.conditions = req.conditions
    model.schedule = req.schedule
    model.actions = req.actions
    model.cooldown_seconds = req.cooldown_seconds

    db.commit()
    db.refresh(model)

    parsed_cond = parse_condition_group(model.conditions)
    actions = [
        RuleAction(
            type=ActionType(a["type"]),
            recipient_group=a.get("recipient_group", "HOUSEHOLD_ADMINS"),
            custom_message_template=a.get("custom_message_template"),
            priority=a.get("priority", "NORMAL")
        )
        for a in model.actions
    ]
    c_rule = CustomerRule(
        id=model.id,
        household_id=model.household_id,
        name=model.name,
        description=model.description or "",
        camera_ids=model.camera_ids or [],
        enabled=model.enabled,
        conditions=parsed_cond,
        actions=actions,
        cooldown_seconds=model.cooldown_seconds
    )
    runtime.event_manager.register_rule(c_rule)

    return model


@router.delete("/{rule_id}")
def delete_rule(
    rule_id: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    stmt = select(CustomerRuleModel).where(CustomerRuleModel.id == rule_id)
    model = db.execute(stmt).scalar_one_or_none()
    if not model:
        raise HTTPException(status_code=404, detail="Rule not found")

    db.delete(model)
    db.commit()
    runtime.event_manager.remove_rule(rule_id)
    return {"status": "deleted"}


@router.post("/simulate")
def simulate_rule(req: RuleSimulateRequest):
    """Simulates rule against a sample event without dispatching live sirens."""
    cond_group = parse_condition_group(req.rule.get("conditions", {}))
    raw_actions = req.rule.get("actions", [])
    actions = [
        RuleAction(
            type=ActionType(a["type"]),
            recipient_group=a.get("recipient_group", "HOUSEHOLD_ADMINS"),
            custom_message_template=a.get("custom_message_template"),
            priority=a.get("priority", "NORMAL")
        )
        for a in raw_actions
    ]
    test_rule = CustomerRule(
        name=req.rule.get("name", "Test Rule"),
        conditions=cond_group,
        actions=actions,
        camera_ids=req.rule.get("camera_ids", []),
        enabled=True
    )

    ev = req.sample_event
    matches = test_rule.matches(ev)
    generated_messages = []
    if matches:
        for action in actions:
            context = {
                "event": {
                    "type": ev.get("event_type"),
                    "camera_id": ev.get("camera_id"),
                    "zone": ev.get("zone_id", "General"),
                    "time": "03:45 PM"
                },
                "metadata": ev.get("metadata", {})
            }
            msg = action.format_message(context, f"Triggered by {ev.get('event_type')}")
            generated_messages.append({"recipient": action.recipient_group, "priority": action.priority, "message": msg})

    return {
        "rule_matches": matches,
        "simulated_notifications": generated_messages
    }
