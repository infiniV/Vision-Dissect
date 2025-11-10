from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ..schemas.responses import DetectionResponse, Detection
from ..models.factory import model_manager
from ..utils.image_utils import validate_and_load_image
from ..config import settings


router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("", response_model=DetectionResponse)
async def detect_objects(
    image: UploadFile = File(...),
    confidence: float = Query(default=settings.yolo_confidence, ge=0.0, le=1.0),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("yolo_detect")
        result = await handler.predict(img, confidence)

        detections = [Detection(**d) for d in result["detections"]]

        return DetectionResponse(detections=detections, count=result["count"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
