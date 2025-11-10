import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

# Try to import optional libraries
try:
    import cv2

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    from ultralytics import YOLO, SAM

    HAS_SAM = True
except Exception:
    HAS_SAM = False


def ensure_viz_dir():
    os.makedirs("../viz", exist_ok=True)


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def rgb_channels(img):
    # img: HxWx3
    return [img[..., i] for i in range(3)]


def grayscale(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def hsv_channels(img):
    pil = Image.fromarray(img)
    hsv = np.array(pil.convert("HSV"))
    return [hsv[..., 0], hsv[..., 1], hsv[..., 2]]


def apply_kernel(img_gray, kernel):
    # Simple 2D convolution using numpy pad and manual sum - works for small kernels
    h, w = img_gray.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img_gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros_like(img_gray, dtype=float)
    for i in range(h):
        for j in range(w):
            region = padded[i : i + kh, j : j + kw]
            out[i, j] = np.sum(region * kernel)
    return out


def sobel_filters(img_gray):
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float)
    gx = apply_kernel(img_gray, kx)
    gy = apply_kernel(img_gray, ky)
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def laplacian(img_gray):
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)
    return apply_kernel(img_gray, k)


def canny_if_available(img_gray):
    if not HAS_CV2:
        return None
    # OpenCV expects uint8
    img_u8 = np.clip(img_gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(img_u8, 100, 200)
    return edges


def compute_depth_midas(img_rgb):
    # Try to run MiDaS via torch hub if available
    try:
        model_type = "MiDaS_small"  # faster
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to("cpu")
        midas.eval()
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        # Ensure img_rgb is a numpy array and convert to PIL properly
        if isinstance(img_rgb, np.ndarray):
            img_pil = Image.fromarray(img_rgb.astype(np.uint8))
        else:
            img_pil = img_rgb

        input_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            prediction = midas(input_tensor)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(
                        img_rgb.shape[:2]
                        if isinstance(img_rgb, np.ndarray)
                        else (img_rgb.height, img_rgb.width)
                    ),
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        # normalize
        depth = (prediction - prediction.min()) / (
            prediction.max() - prediction.min() + 1e-8
        )
        depth = (depth * 255.0).astype(np.uint8)
        return depth
    except Exception as e:
        print("MiDaS depth estimation unavailable:", e)
        return None


def sam_masks_if_available(model_path, img_path, downscale=1):
    if not HAS_SAM:
        return None
    try:
        model = SAM(model_path)
        results = model(img_path, verbose=False)[0]
        # results.masks may be boolean masks at original size - convert to uint8
        if hasattr(results, "masks") and results.masks is not None:
            masks = results.masks.data
            if masks.dtype == torch.bool:
                masks = masks.to(torch.uint8) * 255
            return masks.detach().cpu().numpy()
    except Exception as e:
        print("SAM masks unavailable:", e)
    return None


def visualize_channel_grid(channels, titles, out_path, cmap=None):
    n = len(channels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
    for i, (ch, title) in enumerate(zip(channels, titles)):
        ax = axes[i]
        ax.imshow(ch, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out_path}")


def compare_channels(
    image_path="../test_data/test_image.jpg", sam_model_path="../models/mobile_sam.pt"
):
    ensure_viz_dir()
    img = load_image(image_path)
    small = None
    # use smaller size for faster per-pixel convs
    small_h = 512
    scale = max(img.shape[0] / small_h, 1)
    if scale > 1:
        small = np.array(
            Image.fromarray(img).resize(
                (int(img.shape[1] // scale), int(img.shape[0] // scale)), Image.LANCZOS
            )
        )
    else:
        small = img

    gray = grayscale(small)
    rch = rgb_channels(small)
    hsv = hsv_channels(small)
    gx, gy, sob = sobel_filters(gray)
    lap = laplacian(gray)
    canny = canny_if_available(
        (gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255
    )

    # Visualize RGB channels
    visualize_channel_grid(rch, ["R", "G", "B"], "../viz/channels_rgb.png")

    # Grayscale and gradient/laplacian
    viz_list = [gray, np.abs(gx), np.abs(gy), sob, np.abs(lap)]
    viz_titles = ["Gray", "Sobel-Gx", "Sobel-Gy", "Sobel-Mag", "Laplacian"]
    visualize_channel_grid(viz_list, viz_titles, "../viz/gradients.png", cmap="viridis")

    # HSV
    visualize_channel_grid(hsv, ["H", "S", "V"], "../viz/channels_hsv.png")

    # Canny if available
    if canny is not None:
        visualize_channel_grid(
            [canny], ["Canny Edges"], "../viz/canny.png", cmap="gray"
        )
    else:
        print("OpenCV not available — skipping Canny edge visualization")

    # MiDaS depth if available
    depth = compute_depth_midas(img)
    if depth is not None:
        visualize_channel_grid(
            [depth], ["MiDaS Depth"], "../viz/depth_midas.png", cmap="magma"
        )
    else:
        print("MiDaS not available — depth visualization skipped")

    # SAM masks if available
    masks = sam_masks_if_available(sam_model_path, image_path)
    if masks is not None:
        # masks shape maybe [N, H, W] or [N, 1, H, W] or torch.Size
        if masks.ndim == 3:
            mcount = masks.shape[0]
            sample = [masks[i] for i in range(min(8, mcount))]
            titles = [f"Mask {i}" for i in range(len(sample))]
            visualize_channel_grid(sample, titles, "../viz/sam_masks.png", cmap="gray")
        elif masks.ndim == 4:
            # [N, C, H, W] - flatten channel dim
            N, C, H, W = masks.shape
            sample = [masks[i, 0] for i in range(min(8, N))]
            titles = [f"Mask {i}" for i in range(len(sample))]
            visualize_channel_grid(sample, titles, "../viz/sam_masks.png", cmap="gray")
    else:
        print("SAM not available or no masks produced — skipping mask visualization")


if __name__ == "__main__":
    compare_channels()
