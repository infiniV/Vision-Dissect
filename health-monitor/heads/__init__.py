"""
Model Heads Module
Prediction heads for depth, segmentation, and keypoints
"""

from .pretrained_heads import (
    PretrainedDepthHead,
    PretrainedSegmentationHead,
    PretrainedKeypointsHead
)
from .onnx_heads import (
    ONNXDepthHead,
    ONNXSegmentationHead,
    ONNXKeypointsHead
)

__all__ = [
    'PretrainedDepthHead',
    'PretrainedSegmentationHead',
    'PretrainedKeypointsHead',
    'ONNXDepthHead',
    'ONNXSegmentationHead',
    'ONNXKeypointsHead',
]
