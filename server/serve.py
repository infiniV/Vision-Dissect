from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from PIL import Image
import logging

from .config import settings
from .models.factory import model_manager
from .routers import depth, detection, segmentation, pose
from .schemas.responses import HealthResponse, QueueResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting FastAPI server...")
    logger.info(f"Device: {settings.device}")
    logger.info(f"Max image size: {settings.max_image_size} bytes")

    # Warmup models
    warmup_models = settings.warmup_models_list
    if warmup_models:
        logger.info(f"Warming up models: {warmup_models}")
        dummy_image = Image.new("RGB", (64, 64))

        for model_name in warmup_models:
            try:
                start_time = time.time()

                if model_name == "depthpro":
                    handler = await model_manager.get_model("depthpro")
                    await handler.predict(dummy_image)
                elif model_name == "yolo_detect":
                    handler = await model_manager.get_model("yolo_detect")
                    await handler.predict(dummy_image, confidence=0.5)
                elif model_name == "yolo_segment":
                    handler = await model_manager.get_model("yolo_segment")
                    await handler.predict(dummy_image, confidence=0.5)
                elif model_name == "yolo_pose":
                    handler = await model_manager.get_model("yolo_pose")
                    await handler.predict(dummy_image, confidence=0.5)
                elif model_name.startswith("depth_anything"):
                    variant = (
                        model_name.split("_")[-1]
                        if len(model_name.split("_")) > 2
                        else "vits"
                    )
                    handler = await model_manager.get_model("depth_anything", variant)
                    await handler.predict(dummy_image)
                elif model_name == "sam":
                    handler = await model_manager.get_model("sam")
                    await handler.predict(dummy_image)

                elapsed = time.time() - start_time
                logger.info(f"Warmed up {model_name} in {elapsed:.2f}s")
            except Exception as e:
                logger.error(f"Failed to warm up {model_name}: {e}")

    logger.info("Server startup complete")

    yield

    # Shutdown
    logger.info("Shutting down server...")


app = FastAPI(
    title="Vision Inference Server",
    description="Multi-model inference server for vision tasks",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request size limit middleware
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.max_image_size * 2:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request size exceeds maximum allowed size"},
            )
    return await call_next(request)


# Include routers
app.include_router(depth.router, prefix="/api/v1")
app.include_router(detection.router, prefix="/api/v1")
app.include_router(segmentation.router, prefix="/api/v1")
app.include_router(pose.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        loaded_models=model_manager.get_loaded_models(),
        available_models=[
            "depthpro",
            "depth_anything",
            "yolo_detect",
            "yolo_segment",
            "yolo_pose",
            "sam",
        ],
    )


@app.get("/queue", response_model=QueueResponse)
async def queue_status():
    return QueueResponse(queue_depths=model_manager.get_queue_depths())


@app.get("/")
async def root():
    return {
        "message": "Vision Inference Server",
        "version": "1.0.0",
        "endpoints": {
            "depth": ["/api/v1/depth/depthpro", "/api/v1/depth/depth-anything-v2"],
            "detection": ["/api/v1/detect"],
            "segmentation": ["/api/v1/segment/yolo", "/api/v1/segment/sam"],
            "pose": ["/api/v1/pose"],
            "monitoring": ["/health", "/queue"],
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
