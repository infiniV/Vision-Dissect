from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ..schemas.responses import DepthResponse
from ..models.factory import model_manager
from ..utils.image_utils import validate_and_load_image
from ..utils.response_formatter import format_depth_response
from ..config import settings


router = APIRouter(prefix="/depth", tags=["depth"])


@router.post("/depthpro", response_model=DepthResponse)
async def depth_pro(
    image: UploadFile = File(...),
    format: str = Query(
        default=settings.default_depth_format, regex="^(full|downsampled|stats)$"
    ),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("depthpro")
        result = await handler.predict(img)

        depth_data = format_depth_response(result["depth_map"], format)

        return DepthResponse(
            depth_data=depth_data,
            format=format,
            shape=result["metadata"]["shape"],
            metadata=result["metadata"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")


@router.post("/depth-anything-v2", response_model=DepthResponse)
async def depth_anything_v2(
    image: UploadFile = File(...),
    variant: str = Query(
        default=settings.depth_anything_variant, regex="^(vits|vitb|vitl)$"
    ),
    format: str = Query(
        default=settings.default_depth_format, regex="^(full|downsampled|stats)$"
    ),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("depth_anything", variant)
        result = await handler.predict(img)

        depth_data = format_depth_response(result["depth_map"], format)

        return DepthResponse(
            depth_data=depth_data,
            format=format,
            shape=result["metadata"]["shape"],
            metadata=result["metadata"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
