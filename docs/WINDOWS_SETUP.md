# WINDOWS SETUP GUIDE
## Running Presence B2C Platform on Windows 10/11

---

### Prerequisites
- Python 3.11 or 3.12 (Installed and added to PATH)
- Node.js v18+ (Optional, frontend served directly by FastAPI)

### 1. Run All Tests
To run unit and integration tests across all 10 use cases and API endpoints:
```powershell
python -m pytest ai/tests/test_all_10_plugins.py -v
python -m pytest apps/api/tests/test_api.py -v
```

### 2. Launch Platform Server
```powershell
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Open Web Platform
Open your browser and navigate to:
- **Customer Dashboard**: `http://localhost:8000/`
- **Super Admin Portal**: `http://localhost:8000/admin`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

Default Seeded Accounts:
- **Super Administrator**: `admin@presence.ai` / `admin123`
- **Customer Household**: `demo@presence.ai` / `demo123`
