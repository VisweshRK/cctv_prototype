"""Global AI Runtime Orchestrator for Presence B2C API.
Maintains running master pipelines, event manager, model registry, and websocket connections.
"""

import asyncio
from typing import Dict, List, Optional
from fastapi import WebSocket

from ai.cctv_ai.events import EventManager
from ai.cctv_ai.model_registry import ModelRegistry
from ai.cctv_ai.pipeline import MasterPipeline
from ai.cctv_ai.zones import ZoneManager


class RuntimeOrchestrator:
    def __init__(self):
        self.event_manager = EventManager()
        self.zone_manager = ZoneManager("e:/cctv/ai/zones.yaml")
        self.model_registry = ModelRegistry()
        self.pipelines: Dict[str, MasterPipeline] = {}
        self.active_websockets: List[WebSocket] = []

        # Wire event manager broadcasts to connected websockets
        self.event_manager.subscribe(self._broadcast_to_websockets)

    def get_or_create_pipeline(self, camera_id: str, stream_source: str = "synthetic") -> MasterPipeline:
        if camera_id not in self.pipelines:
            p = MasterPipeline(
                camera_id=camera_id,
                stream_source=stream_source,
                event_manager=self.event_manager,
                zone_manager=self.zone_manager
            )
            p.start()
            self.pipelines[camera_id] = p
        return self.pipelines[camera_id]

    async def connect_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_websockets.append(websocket)

    def disconnect_websocket(self, websocket: WebSocket):
        if websocket in self.active_websockets:
            self.active_websockets.remove(websocket)

    def _broadcast_to_websockets(self, msg: dict) -> None:
        if not self.active_websockets:
            return
        try:
            loop = asyncio.get_running_loop()
            for ws in list(self.active_websockets):
                loop.create_task(ws.send_json(msg))
        except RuntimeError:
            pass  # No running event loop in current thread


runtime = RuntimeOrchestrator()
