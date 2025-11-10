import asyncio
import threading
from typing import Dict, Any, Optional
from collections import defaultdict


class ModelManager:
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _model_locks: Dict[str, asyncio.Lock] = {}
    _queue_depths: Dict[str, int] = defaultdict(int)

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    async def get_model(self, model_type: str, variant: Optional[str] = None):
        key = f"{model_type}_{variant}" if variant else model_type

        if key not in self._model_locks:
            self._model_locks[key] = asyncio.Lock()

        self._queue_depths[key] += 1

        try:
            async with self._model_locks[key]:
                if key not in self._models:
                    self._models[key] = await self._load_model(model_type, variant)
                return self._models[key]
        finally:
            self._queue_depths[key] -= 1

    async def _load_model(self, model_type: str, variant: Optional[str] = None):
        if model_type == "depthpro":
            from .depth_handler import DepthProHandler

            handler = DepthProHandler()
            await handler.load()
            return handler

        elif model_type == "depth_anything":
            from .depth_handler import DepthAnythingHandler

            handler = DepthAnythingHandler(variant or "vits")
            await handler.load()
            return handler

        elif model_type == "yolo_detect":
            from .yolo_handler import YOLODetectionHandler

            handler = YOLODetectionHandler()
            await handler.load()
            return handler

        elif model_type == "yolo_segment":
            from .yolo_handler import YOLOSegmentationHandler

            handler = YOLOSegmentationHandler()
            await handler.load()
            return handler

        elif model_type == "yolo_pose":
            from .yolo_handler import YOLOPoseHandler

            handler = YOLOPoseHandler()
            await handler.load()
            return handler

        elif model_type == "sam":
            from .sam_handler import SAMHandler

            handler = SAMHandler()
            await handler.load()
            return handler

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_queue_depths(self) -> Dict[str, int]:
        return dict(self._queue_depths)

    def get_loaded_models(self) -> list:
        return list(self._models.keys())


model_manager = ModelManager()
