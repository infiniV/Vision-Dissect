from pydantic import BaseModel
from typing import List, Union, Dict, Any


class DepthResponse(BaseModel):
    depth_data: Union[str, Dict[str, Any]]
    format: str
    shape: List[int]
    metadata: Dict[str, Any]


class Detection(BaseModel):
    bbox: List[float]
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    detections: List[Detection]
    count: int


class SegmentationResponse(BaseModel):
    detections: List[Detection]
    masks: List[str]
    count: int
    format: str


class Keypoint(BaseModel):
    name: str
    x: float
    y: float
    confidence: float


class Pose(BaseModel):
    person_id: int
    bbox: List[float]
    confidence: float
    keypoints: List[Keypoint]


class PoseResponse(BaseModel):
    poses: List[Pose]
    count: int


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
    available_models: List[str]


class QueueResponse(BaseModel):
    queue_depths: Dict[str, int]
