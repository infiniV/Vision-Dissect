import torch
import numpy as np
from PIL import Image
from typing import Dict, Any
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))


class DepthProHandler:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None

    async def load(self):
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

        self.device = torch.device("cpu")
        self.processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(
            self.device
        )
        self.model.eval()

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        post_processed = self.processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )

        depth = post_processed[0]["predicted_depth"].cpu().numpy()
        fov = post_processed[0]["field_of_view"].item()
        focal_length = post_processed[0]["focal_length"].item()

        return {
            "depth_map": depth,
            "metadata": {
                "field_of_view": fov,
                "focal_length": focal_length,
                "min_depth": float(depth.min()),
                "max_depth": float(depth.max()),
                "shape": list(depth.shape),
            },
        }


class DepthAnythingHandler:
    def __init__(self, variant: str = "vits"):
        self.variant = variant
        self.model = None
        self.device = None
        self.model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
            "vitb": {
                "encoder": "vitb",
                "features": 128,
                "out_channels": [96, 192, 384, 768],
            },
            "vitl": {
                "encoder": "vitl",
                "features": 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        }

    async def load(self):
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = torch.device("cpu")

        config = self.model_configs.get(self.variant)
        if not config:
            raise ValueError(f"Unknown variant: {self.variant}")

        self.model = DepthAnythingV2(**config)

        # Download weights from HuggingFace
        variant_map = {"vits": "Small", "vitb": "Base", "vitl": "Large"}
        model_name = variant_map[self.variant]
        url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{model_name}/resolve/main/depth_anything_v2_{self.variant}.pth"

        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        import cv2

        # Convert PIL to OpenCV format
        raw_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            depth = self.model.infer_image(raw_image, input_size=518)

        return {
            "depth_map": depth,
            "metadata": {
                "min_depth": float(depth.min()),
                "max_depth": float(depth.max()),
                "shape": list(depth.shape),
            },
        }
