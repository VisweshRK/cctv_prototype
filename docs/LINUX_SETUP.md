# LINUX SETUP GUIDE
## Running Presence B2C Platform on Ubuntu / Debian / Rocky Linux

---

### 1. Install System Dependencies
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg libgl1-mesa-glx python3-pip
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
python3 -m pytest ai/tests/test_all_10_plugins.py -v
python3 -m pytest apps/api/tests/test_api.py -v
```

### 4. Run Server
```bash
uvicorn apps.api.app.main:app --host 0.0.0.0 --port 8000
```
