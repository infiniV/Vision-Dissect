import numpy as np
import base64
import io
from PIL import Image
from typing import Union, Dict, Any


def format_depth_response(
    depth: np.ndarray, format: str = "downsampled"
) -> Union[str, Dict[str, Any]]:
    if format == "stats":
        return {
            "min": float(depth.min()),
            "max": float(depth.max()),
            "mean": float(depth.mean()),
            "median": float(np.median(depth)),
            "percentiles": {
                "25": float(np.percentile(depth, 25)),
                "50": float(np.percentile(depth, 50)),
                "75": float(np.percentile(depth, 75)),
                "90": float(np.percentile(depth, 90)),
                "95": float(np.percentile(depth, 95)),
            },
        }

    elif format == "downsampled":
        # Downsample to max 512x512
        h, w = depth.shape
        if h > 512 or w > 512:
            scale = 512 / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            depth_img = Image.fromarray(depth)
            depth_img = depth_img.resize((new_w, new_h), Image.BILINEAR)
            depth = np.array(depth_img)

        return _array_to_base64(depth)

    else:  # full
        return _array_to_base64(depth)


def format_mask_response(masks: np.ndarray, format: str = "downsampled") -> list:
    if masks is None:
        return []

    mask_list = []
    for mask in masks:
        if format == "downsampled":
            # Downsample to max 512x512
            if len(mask.shape) == 3:
                mask = mask[0]

            h, w = mask.shape
            if h > 512 or w > 512:
                scale = 512 / max(h, w)
                new_h = int(h * scale)
                new_w = int(w * scale)
                mask_img = Image.fromarray((mask * 255).astype(np.uint8))
                mask_img = mask_img.resize((new_w, new_h), Image.NEAREST)
                mask = np.array(mask_img) / 255.0

        mask_list.append(_array_to_base64(mask))

    return mask_list


def _array_to_base64(arr: np.ndarray) -> str:
    # Normalize to 0-255
    arr_normalized = (
        (arr - arr.min()) / (arr.max() - arr.min()) if arr.max() > arr.min() else arr
    )
    arr_uint8 = (arr_normalized * 255).astype(np.uint8)

    # Convert to PIL Image
    img = Image.fromarray(arr_uint8)

    # Encode to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return img_str
