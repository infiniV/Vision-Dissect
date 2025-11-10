from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ..schemas.responses import SegmentationResponse, Detection
from ..models.factory import model_manager
from ..utils.image_utils import validate_and_load_image
from ..utils.response_formatter import format_mask_response
from ..config import settings


router = APIRouter(prefix="/segment", tags=["segmentation"])


@router.post("/yolo", response_model=SegmentationResponse)
async def segment_yolo(
    image: UploadFile = File(...),
    confidence: float = Query(default=settings.yolo_confidence, ge=0.0, le=1.0),
    format: str = Query(default="downsampled", regex="^(full|downsampled)$"),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("yolo_segment")
        result = await handler.predict(img, confidence)

        detections = [Detection(**d) for d in result["detections"]]
        masks = (
            format_mask_response(result["masks"], format)
            if result["masks"] is not None
            else []
        )

        return SegmentationResponse(
            detections=detections, masks=masks, count=result["count"], format=format
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")


@router.post("/sam", response_model=SegmentationResponse)
async def segment_sam(
    image: UploadFile = File(...),
    format: str = Query(default="downsampled", regex="^(full|downsampled)$"),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("sam")
        result = await handler.predict(img)

        masks = (
            format_mask_response(result["masks"], format)
            if result["masks"] is not None
            else []
        )

        return SegmentationResponse(
            detections=[], masks=masks, count=result["count"], format=format
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
