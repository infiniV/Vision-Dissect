from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from ..schemas.responses import PoseResponse, Pose, Keypoint
from ..models.factory import model_manager
from ..utils.image_utils import validate_and_load_image
from ..config import settings


router = APIRouter(prefix="/pose", tags=["pose"])


@router.post("", response_model=PoseResponse)
async def detect_pose(
    image: UploadFile = File(...),
    confidence: float = Query(default=settings.yolo_confidence, ge=0.0, le=1.0),
):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)

        handler = await model_manager.get_model("yolo_pose")
        result = await handler.predict(img, confidence)

        poses = []
        for pose_data in result["poses"]:
            keypoints = [Keypoint(**kp) for kp in pose_data["keypoints"]]
            poses.append(
                Pose(
                    person_id=pose_data["person_id"],
                    bbox=pose_data["bbox"],
                    confidence=pose_data["confidence"],
                    keypoints=keypoints,
                )
            )

        return PoseResponse(poses=poses, count=result["count"])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
