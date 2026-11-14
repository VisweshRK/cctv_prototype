"""Main FastAPI Application Entrypoint for Presence B2C AI CCTV Platform.
"""

from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from apps.api.app.api.admin import router as admin_router
from apps.api.app.api.alerts import router as alerts_router
from apps.api.app.api.analytics import router as analytics_router
from apps.api.app.api.auth import router as auth_router
from apps.api.app.api.cameras import router as cameras_router
from apps.api.app.api.events import router as events_router
from apps.api.app.api.models import router as models_router
from apps.api.app.api.rules import router as rules_router
from apps.api.app.api.stream import router as stream_router
from apps.api.app.api.zones import router as zones_router

from apps.api.app.core.config import settings
from apps.api.app.core.init_db import init_database
from apps.api.app.core.runtime import runtime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("presence-b2c")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize tables and seed default accounts
    logger.info("Initializing database...")
    init_database()

    # Pre-load customer rules into runtime event manager
    from apps.api.app.api.rules import load_all_rules_into_runtime
    load_all_rules_into_runtime()
    logger.info(f"Loaded {len(runtime.event_manager.rules)} active customer rules into runtime.")

    # Initialize default camera pipeline if not running test suite
    if os.getenv("TESTING") != "1":
        runtime.get_or_create_pipeline("camera_main", "synthetic")
        logger.info("Default AI CCTV MasterPipeline running.")

    yield

    # Shutdown
    logger.info("Shutting down AI pipelines...")
    for p in runtime.pipelines.values():
        p.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(auth_router, prefix=settings.API_PREFIX)
app.include_router(cameras_router, prefix=settings.API_PREFIX)
app.include_router(rules_router, prefix=settings.API_PREFIX)
app.include_router(events_router, prefix=settings.API_PREFIX)
app.include_router(alerts_router, prefix=settings.API_PREFIX)
app.include_router(zones_router, prefix=settings.API_PREFIX)
app.include_router(models_router, prefix=settings.API_PREFIX)
app.include_router(analytics_router, prefix=settings.API_PREFIX)
app.include_router(admin_router, prefix=settings.API_PREFIX)
app.include_router(stream_router, prefix=settings.API_PREFIX)


# Real-time WebSockets endpoint for live notifications and event streaming
@app.websocket("/ws/live")
async def websocket_live_feed(websocket: WebSocket):
    await runtime.connect_websocket(websocket)
    try:
        while True:
            # Keep-alive receive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        runtime.disconnect_websocket(websocket)
    except Exception:
        runtime.disconnect_websocket(websocket)


# Serve Single Page Application
@app.get("/", response_class=FileResponse)
def serve_index():
    dist_file = "e:/cctv/apps/web/dist/index.html"
    if os.path.exists(dist_file):
        return FileResponse(dist_file)
    return FileResponse("e:/cctv/apps/web/public/index.html")


@app.get("/admin", response_class=FileResponse)
def serve_admin():
    dist_file = "e:/cctv/apps/web/dist/index.html"
    if os.path.exists(dist_file):
        return FileResponse(dist_file)
    return FileResponse("e:/cctv/apps/web/public/index.html")


# Mount static assets if web build exists
web_dist_assets = "e:/cctv/apps/web/dist/assets"
if os.path.exists(web_dist_assets):
    app.mount("/assets", StaticFiles(directory=web_dist_assets), name="assets")

web_public = "e:/cctv/apps/web/public"
if os.path.exists(web_public):
    app.mount("/public", StaticFiles(directory=web_public), name="public")
