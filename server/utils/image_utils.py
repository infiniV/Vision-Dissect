from PIL import Image
import io
from fastapi import UploadFile, HTTPException


async def validate_and_load_image(file: UploadFile, max_size: int) -> Image.Image:
    # Check file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    file_ext = file.filename.lower().split(".")[-1] if file.filename else ""

    if f".{file_ext}" not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported: jpg, jpeg, png, webp",
        )

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > max_size:
        raise HTTPException(
            status_code=413, detail=f"Image size exceeds maximum of {max_size} bytes"
        )

    # Load image
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
