
import os
import cv2
import time
import threading

from datetime import datetime
from collections import deque
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL", "").strip()
MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.45"))
EVENT_COOLDOWN = int(os.getenv("EVENT_COOLDOWN", "10"))

# Restricted area: normalized frame coordinates.
# Adjust these four corners to match your camera view.
ZONE = [
    (0.25, 0.20),
    (0.75, 0.20),
    (0.75, 0.90),
    (0.25, 0.90)
]

# Keep recent events in memory.
events = deque(maxlen=200)

# ==========================================
# CAMERA AND DETECTION ENGINE
# ==========================================

class IntrusionDetector:

    def __init__(self):

        self.model = None
        self.cap = None

        self.frame = None
        self.lock = threading.Lock()

        self.running = False
        self.connected = False
        self.model_loaded = False

        self.thread = None
        self.error = None

        self.person_count = 0
        self.last_event_time = 0

    def load_model(self):

        print("Loading YOLO model...")

        self.model = YOLO(MODEL_PATH)

        self.model_loaded = True

        print("YOLO model loaded successfully.")

    def connect_camera(self):

        if not RTSP_URL:
            self.error = "RTSP_URL missing in .env"
            return False

        print("Connecting to RTSP camera...")

        self.cap = cv2.VideoCapture(
            RTSP_URL,
            cv2.CAP_FFMPEG
        )

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            self.connected = False
            self.error = "Unable to open RTSP stream"

            return False

        self.connected = True
        self.error = None

        print("RTSP camera connected.")

        return True

    def create_event(self, count):

        current_time = time.time()

        if current_time - self.last_event_time < EVENT_COOLDOWN:
            return

        self.last_event_time = current_time

        event = {
            "id": int(current_time * 1000),
            "type": "INTRUSION",
            "message": "Person detected inside restricted zone",
            "person_count": count,
            "timestamp": datetime.now().isoformat()
        }

        events.appendleft(event)

        print("INTRUSION ALERT:", event)

    def process_frame(self, frame):

        height, width = frame.shape[:2]

        polygon = [
            (int(x * width), int(y * height))
            for x, y in ZONE
        ]

        # Draw restricted area.
        cv2.polylines(
            frame,
            [__import__("numpy").array(polygon)],
            True,
            (0, 255, 255),
            2
        )

        results = self.model.predict(
            frame,
            classes=[0],       # Person class only
            conf=CONFIDENCE,
            imgsz=640,
            verbose=False
        )

        intrusion_count = 0
        total_persons = 0

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                total_persons += 1

                inside_zone = (
                    cv2.pointPolygonTest(
                        __import__("numpy").array(polygon),
                        (center_x, center_y),
                        False
                    ) >= 0
                )

                if inside_zone:

                    intrusion_count += 1

                    color = (0, 0, 255)

                    self.create_event(intrusion_count)

                    label = "INTRUSION"

                else:

                    color = (0, 255, 0)
                    label = "PERSON"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        cv2.putText(
            frame,
            f"Persons: {total_persons}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Intrusions: {intrusion_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        self.person_count = total_persons

        return frame

    def capture_loop(self):

        while self.running:

            try:

                if self.cap is None or not self.cap.isOpened():

                    self.connected = False

                    if not self.connect_camera():

                        time.sleep(3)
                        continue

                success, frame = self.cap.read()

                if not success or frame is None:

                    self.connected = False
                    self.error = "Camera frame unavailable"

                    self.cap.release()
                    self.cap = None

                    time.sleep(2)
                    continue

                processed = self.process_frame(frame)

                with self.lock:

                    self.frame = processed.copy()

                self.connected = True
                self.error = None

            except Exception as error:

                self.error = str(error)

                print("Detection error:", error)

                self.connected = False

                if self.cap is not None:

                    self.cap.release()
                    self.cap = None

                time.sleep(2)

    def start(self):

        if self.running:
            return

        if not self.model_loaded:
            raise RuntimeError("YOLO model is not loaded")

        self.running = True

        self.thread = threading.Thread(
            target=self.capture_loop,
            daemon=True
        )

        self.thread.start()

    def get_frame(self):

        with self.lock:

            if self.frame is None:
                return None

            return self.frame.copy()

    def stop(self):

        self.running = False

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.connected = False

        if self.thread and self.thread.is_alive():

            self.thread.join(timeout=3)

# ==========================================
# FASTAPI APPLICATION
# ==========================================

detector = IntrusionDetector()

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:

        detector.load_model()
        detector.start()

    except Exception as error:

        detector.error = str(error)
        print("Startup error:", error)

    yield

    detector.stop()

app = FastAPI(
    title="AI Intrusion Detection System",
    version="1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==========================================
# API ROUTES
# ==========================================

@app.get("/")
def home():

    return {
        "message": "AI Intrusion Detection Backend Running",
        "model_loaded": detector.model_loaded
    }

@app.get("/api/health")
def health():

    return {
        "status": "online",
        "model_loaded": detector.model_loaded
    }

@app.get("/api/camera/status")
def camera_status():

    return {
        "connected": detector.connected,
        "model_loaded": detector.model_loaded,
        "person_count": detector.person_count,
        "error": detector.error
    }

@app.get("/api/camera/stream")
def camera_stream():

    def generate():

        while True:

            frame = detector.get_frame()

            if frame is None:

                time.sleep(0.1)
                continue

            success, buffer = cv2.imencode(
                ".jpg",
                frame
            )

            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/events")
def get_events():

    return list(events)

@app.post("/api/camera/reconnect")
def reconnect():

    detector.stop()

    if not detector.model_loaded:

        return JSONResponse(
            status_code=503,
            content={"error": "YOLO model is not loaded"}
        )

    detector.start()

    return {
        "message": "Camera reconnection requested"
    }