# Presence B2C — AI CCTV Monitoring Platform
### 10 AI Use Cases + Dynamic Customer Rules + Monitoring Website

Presence B2C is an end-to-end AI CCTV monitoring platform engineered for consumer homes and businesses. It integrates 10 AI safety use cases into a single-decode shared pipeline, features a typed customer dynamic rule builder with customizable notification templates (e.g. child arrival from school), and provides a responsive dashboard with dedicated **Customer Dashboard** and **Super Admin Portal** views.

---

## 1. Quick Start

### 1.1. Run All Tests
```bash
python -m pytest ai/tests/test_all_10_plugins.py -v
python -m pytest apps/api/tests/test_api.py -v
```

### 1.2. Start the Server
```bash
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000
```

### 1.3. Open the Monitoring Website
Open `http://localhost:8000/` in your web browser.
- **Customer Dashboard**: `http://localhost:8000/`
- **Super Admin Page**: `http://localhost:8000/admin` (or toggle the top header switcher)
- **Interactive API Documentation**: `http://localhost:8000/docs`

---

## 2. Seeded Accounts
- **Super Administrator**: `admin@presence.ai` / password: `admin123`
- **Customer Household**: `demo@presence.ai` / password: `demo123`

---

## 3. The 10 AI Use Cases
1. **Intrusion Detection**: Armed entrance monitoring with entry/exit tracking.
2. **Camera Tampering**: Detects lens covering, blur, displacement, and freeze.
3. **Facial Recognition**: Consent biometrics; enables custom rules like *"Child came from school at 3 PM"*.
4. **Restricted Area Breach**: Polygon zones and loitering timers.
5. **Fall & Person-Down Detection**: Bounding-box aspect dynamics + motionless ground persistence.
6. **Room Occupancy Analytics**: Headcount tracking with temporal occlusion debounce.
7. **Electricity During Absence**: Identifies appliances running while room is empty; estimates avoidable kWh and cost.
8. **Fire & Smoke Detection**: Kitchen cooking flame awareness vs prohibited room fires.
9. **Vehicle & Number Plate Log**: Tracks cars/motorcycles and logs Indian HSRP plates.
10. **Baby Crying Monitor**: Acoustic 16 kHz RMS and frequency harmonic classifier with episode grouping.

---

## 4. Documentation
- [Architecture & Data Pipeline](docs/ARCHITECTURE.md)
- [Camera Hardware Diagnostics](docs/CAMERA_COMPATIBILITY.md)
- [Dynamic Rule Engine Guide](docs/RULE_ENGINE.md)
- [Commercial AI Model Provenance](docs/AI_MODELS.md)
- [Windows Setup](docs/WINDOWS_SETUP.md)
- [Linux Setup](docs/LINUX_SETUP.md)
- [Handoff & Verification Report](HANDOFF.md)
