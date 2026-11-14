# DYNAMIC CUSTOMER RULE ENGINE SPECIFICATION
## Section 15 Architecture & DSL Reference

---

## 1. Overview

The Presence B2C Dynamic Rule Engine enables non-technical customers to configure event-driven rules for all 10 CCTV AI use cases without writing code.

---

## 2. Rule DSL Schema

Rules are defined in typed JSON.
To prevent security vulnerabilities (such as command injection, arbitrary Python eval, or SQL injection), all condition queries must reference allowlisted fields:

### Allowlisted Fields:
- `event.type`
- `event.severity`
- `event.camera_id`
- `event.zone_id`
- `event.confidence`
- `event.module_id`
- `metadata.person_name`
- `metadata.is_known`
- `metadata.count`
- `metadata.plate_number`
- `metadata.appliance_type`
- `metadata.duration_seconds`
- `metadata.fall_type`
- `metadata.is_fire`
- `metadata.is_smoke`

### Supported Operators:
- `EQ`, `NEQ`
- `GT`, `GTE`, `LT`, `LTE`
- `CONTAINS`
- `IN`, `NOT_IN`
- `STARTS_WITH`

---

## 3. Sample Rule: Child Returned From School

```json
{
  "name": "Child Came From School Notification",
  "camera_ids": ["camera_main"],
  "enabled": true,
  "conditions": {
    "operator": "AND",
    "items": [
      {
        "field": "event.type",
        "operator": "EQ",
        "value": "ENROLLED_PERSON_RECOGNIZED"
      },
      {
        "field": "metadata.person_name",
        "operator": "EQ",
        "value": "Aarav Sharma"
      }
    ]
  },
  "schedule": {
    "timezone": "Asia/Kolkata",
    "start_time": "14:00",
    "end_time": "18:00",
    "days_of_week": [0, 1, 2, 3, 4]
  },
  "actions": [
    {
      "type": "PUSH_NOTIFICATION",
      "recipient_group": "HOUSEHOLD_ADMINS",
      "custom_message_template": "Welcome Home! Child {metadata.person_name} came from school at {event.time}!",
      "priority": "HIGH"
    }
  ],
  "cooldown_seconds": 300
}
```

When evaluated, this rule produces:
`"Welcome Home! Child Aarav Sharma came from school at 03:45 PM!"`
