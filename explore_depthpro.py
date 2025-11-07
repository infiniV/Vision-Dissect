import requests
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


def load_image(url=None, image_path=None):
    """Load image from URL or local path."""
    if url:
        print(f"Loading image from URL...")
        image = Image.open(requests.get(url, stream=True).raw)
    elif image_path:
        print(f"Loading image from {image_path}...")
        image = Image.open(image_path)
    else:
        raise ValueError("Provide either url or image_path")

    print(f"  Image size: {image.size}")
    return image


def extract_depth(image, model, image_processor, device):
    """Run depth estimation on image."""
    print("Processing depth estimation...")

    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    field_of_view = post_processed_output[0]["field_of_view"]
    focal_length = post_processed_output[0]["focal_length"]
    depth = post_processed_output[0]["predicted_depth"]

    print(f"  Field of view: {field_of_view:.2f}")
    print(f"  Focal length: {focal_length:.2f}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth range: [{depth.min():.2f}, {depth.max():.2f}]")

    return depth, field_of_view, focal_length


def normalize_depth(depth):
    """Normalize depth map to 0-255 range."""
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    depth = depth * 255.0
    depth = depth.detach().cpu().numpy()
    return depth


def visualize_depth(
    image, depth, field_of_view, focal_length, save_path="viz/depth_estimation.png"
):
    """Visualize original image and depth map side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")

    # Depth map (viridis colormap)
    depth_normalized = normalize_depth(depth)
    im1 = axes[1].imshow(depth_normalized, cmap="viridis")
    axes[1].set_title("Depth Map (Viridis)", fontsize=14)
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Depth map (turbo colormap)
    im2 = axes[2].imshow(depth_normalized, cmap="turbo")
    axes[2].set_title("Depth Map (Turbo)", fontsize=14)
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"DepthPro Estimation | FOV: {field_of_view:.2f} | Focal Length: {focal_length:.2f}",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def visualize_depth_slices(depth, num_slices=8, save_path="viz/depth_slices.png"):
    """Visualize depth map at different threshold slices."""
    depth_normalized = normalize_depth(depth)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i in range(num_slices):
        threshold = (i + 1) / (num_slices + 1) * 255
        binary_slice = (depth_normalized > threshold).astype(float)

        axes[i].imshow(binary_slice, cmap="gray")
        axes[i].set_title(f"Depth > {threshold:.0f}", fontsize=10)
        axes[i].axis("off")

    plt.suptitle("Depth Threshold Slices", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print("\nLoading DepthPro model...")
    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    print("  Model loaded successfully")

    # Load image (use URL or local path)
    url = "https://huggingface.co/datasets/mishig/sample_images/resolve/main/tiger.jpg"
    # image = load_image(url=url)
    image = load_image(image_path="test_image.jpg")

    # Extract depth
    depth, field_of_view, focal_length = extract_depth(
        image, model, image_processor, device
    )

    # Visualize results
    print("\nGenerating visualizations...")
    visualize_depth(image, depth, field_of_view, focal_length)
    visualize_depth_slices(depth, num_slices=8)

    print("\nDone!")


if __name__ == "__main__":
    main()
