from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    device: str = "cpu"
    max_image_size: int = 10485760  # 10MB

    depth_anything_variant: str = "vits"
    yolo_confidence: float = 0.5
    use_onnx: bool = True
    default_depth_format: str = "downsampled"

    warmup_models: str = "depthpro,yolo_detect"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), ".env")
        case_sensitive = False

    @property
    def warmup_models_list(self) -> List[str]:
        if not self.warmup_models:
            return []
        return [m.strip() for m in self.warmup_models.split(",")]


settings = Settings()
