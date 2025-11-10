# Adding New Models to the Inference Server

## Architecture Overview

The server follows a factory pattern with clear separation of concerns:

- `models/factory.py` - Central model registry and lazy loading
- `models/*_handler.py` - Model-specific loading and inference logic
- `routers/*.py` - FastAPI endpoints for each category
- `schemas/responses.py` - Pydantic models for API responses
- `utils/` - Shared utilities for image processing and response formatting

## Step-by-Step Guide

### 1. Create Model Handler

Create a new file in `server/models/` (e.g., `my_model_handler.py`):

```python
from PIL import Image
from typing import Dict, Any

class MyModelHandler:
    def __init__(self, variant: str = "default"):
        self.variant = variant
        self.model = None

    async def load(self):
        # Load your model here
        # Example: self.model = load_model_from_path()
        pass

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        # Run inference
        # Return structured dict with results
        return {
            "result": "your_output",
            "metadata": {}
        }
```

### 2. Register in Factory

Add to `models/factory.py` in the `_load_model` method:

```python
elif model_type == "my_model":
    from .my_model_handler import MyModelHandler
    handler = MyModelHandler(variant or "default")
    await handler.load()
    return handler
```

### 3. Create Pydantic Schema

Add to `schemas/responses.py`:

```python
class MyModelResponse(BaseModel):
    result: str
    metadata: Dict[str, Any]
```

### 4. Create Router

Create `routers/my_model.py`:

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from ..schemas.responses import MyModelResponse
from ..models.factory import model_manager
from ..utils.image_utils import validate_and_load_image
from ..config import settings

router = APIRouter(prefix="/my-model", tags=["my-model"])

@router.post("", response_model=MyModelResponse)
async def predict_my_model(image: UploadFile = File(...)):
    try:
        img = await validate_and_load_image(image, settings.max_image_size)
        handler = await model_manager.get_model("my_model")
        result = await handler.predict(img)

        return MyModelResponse(
            result=result["result"],
            metadata=result["metadata"]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model inference failed: {str(e)}")
```

### 5. Register Router

Add to `serve.py`:

```python
from .routers import my_model

app.include_router(my_model.router, prefix="/api/v1")
```

### 6. Update Configuration (Optional)

If your model needs configuration, add to `.env`:

```
MY_MODEL_PARAM=value
```

And add to `config.py`:

```python
my_model_param: str = "value"
```

### 7. Add to Warmup (Optional)

To warm up your model on startup, add to `.env`:

```
WARMUP_MODELS=depthpro,yolo_detect,my_model
```

And add warmup logic to `serve.py` lifespan function:

```python
elif model_name == "my_model":
    handler = await model_manager.get_model("my_model")
    await handler.predict(dummy_image)
```

## Examples

### DepthPro Handler

```python
class DepthProHandler:
    async def load(self):
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
        self.processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf")
        self.model.eval()

    async def predict(self, image: Image.Image) -> Dict[str, Any]:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return {"depth_map": outputs.depth, "metadata": {...}}
```

### YOLO Detection Handler

```python
class YOLODetectionHandler:
    async def load(self):
        self.session = ort.InferenceSession("models/yolo11n.onnx")

    async def predict(self, image: Image.Image, confidence: float = 0.5):
        img_array = self._preprocess(image)
        outputs = self.session.run(None, {self.input_name: img_array})
        detections = self._postprocess(outputs[0], confidence)
        return {"detections": detections, "count": len(detections)}
```

## Best Practices

1. Always implement `async load()` and `async predict()` methods
2. Use structured dicts for return values
3. Handle exceptions in handlers and let routers convert to HTTP errors
4. Keep preprocessing/postprocessing logic in handlers
5. Use utils for shared functionality (image validation, response formatting)
6. Add proper type hints
7. Follow existing naming conventions

## Testing

Test your endpoint with curl:

```bash
curl -X POST "http://localhost:8000/api/v1/my-model" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@test.jpg"
```

Or use the automatic docs at `http://localhost:8000/docs`
