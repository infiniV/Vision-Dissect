from ultralytics import SAM
from PIL import Image
from typing import Dict, Any
import torch
import os


class SAMHandler:
    def __init__(self):
        self.model = None

    async def load(self):
        model_path = os.path.join(
            os.path.dirname(__file__), "../../models/mobile_sam.pt"
        )
        self.model = SAM(model_path)

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        results = self.model(image, verbose=False)[0]

        masks = None
        if hasattr(results, "masks") and results.masks is not None:
            masks = results.masks.data.cpu().numpy()

        return {"masks": masks, "count": len(masks) if masks is not None else 0}
