"""
Depth Visualization App
Interactive visualization of depth estimation following Depth Anything V2 conventions
"""

import torch
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_depth_model(encoder="vits", device="cuda"):
    """Load Depth Anything V2 model."""
    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
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

        if encoder not in model_configs:
            print(f"Invalid encoder: {encoder}. Using 'vits' instead.")
            encoder = "vits"

        print(f"Loading Depth Anything V2 ({encoder})...")
        model = DepthAnythingV2(**model_configs[encoder])
        model = model.to(device).eval()
        return model

    except ImportError:
        print("\nError: depth_anything_v2 module not found.")
        print("\nPlease clone and setup the Depth-Anything-V2 repository:")
        print("  git clone https://github.com/DepthAnything/Depth-Anything-V2")
        print("  Copy the depth_anything_v2 folder to the project root")
        return None


def infer_depth(model, image_path, input_size=518):
    """
    Infer depth from image.

    Args:
        model: Depth Anything V2 model
        image_path: Path to input image
        input_size: Input size for the model

    Returns:
        raw_image: Original BGR image
        depth: Raw depth map (HxW numpy array)
    """
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        raise ValueError(f"Could not load image from {image_path}")

    print(f"Image shape: {raw_image.shape}")

    with torch.no_grad():
        depth = model.infer_image(raw_image, input_size)

    print(f"Depth shape: {depth.shape}")
    print(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")

    return raw_image, depth


def apply_colormap(depth, cmap_name="Spectral_r", grayscale=False):
    """
    Apply colormap to depth following official Depth Anything V2 method.

    Args:
        depth: Raw depth map (HxW numpy array)
        cmap_name: Matplotlib colormap name
        grayscale: If True, return grayscale depth instead

    Returns:
        depth_normalized: Normalized depth (0-255)
        depth_colored: Colored depth in BGR format
    """
    # Normalize depth to 0-255
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)

    if grayscale:
        # Return grayscale depth as 3-channel for consistency
        depth_colored = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
    else:
        # Apply colormap
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

    return depth_normalized, depth_colored


def create_side_by_side(raw_image, depth_colored, margin_width=50):
    """
    Create side-by-side comparison like official Depth Anything V2.

    Args:
        raw_image: Original image in BGR
        depth_colored: Colored depth map in BGR
        margin_width: Width of white margin between images

    Returns:
        combined: Side-by-side image
    """
    # Ensure depth is same size as original
    if raw_image.shape[:2] != depth_colored.shape[:2]:
        depth_colored = cv2.resize(
            depth_colored, (raw_image.shape[1], raw_image.shape[0])
        )

    # Create white margin
    split_region = np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255

    # Concatenate
    combined = cv2.hconcat([raw_image, split_region, depth_colored])

    return combined


def visualize_all_colormaps(depth, output_dir=None):
    """
    Visualize depth with different colormaps for comparison.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "viz"

    # Common colormaps used in depth visualization
    colormaps = [
        "Spectral_r",  # Official Depth Anything V2
        "viridis",  # Good perceptual uniformity
        "plasma",  # High contrast
        "magma",  # Dark background
        "inferno",  # Fire-like
        "turbo",  # Rainbow-like
        "jet",  # Traditional (not recommended)
        "gray",  # Grayscale
    ]

    # Normalize once
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

    # Create figure
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for idx, cmap_name in enumerate(colormaps):
        ax = axes[idx]
        im = ax.imshow(depth_normalized, cmap=cmap_name)

        # Highlight the official one
        if cmap_name == "Spectral_r":
            ax.set_title(
                f"{cmap_name} (Official)", fontsize=12, weight="bold", color="red"
            )
        else:
            ax.set_title(cmap_name, fontsize=12, weight="bold")

        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Depth Map Visualization - Colormap Comparison", fontsize=16, weight="bold"
    )
    plt.tight_layout()

    if output_dir:
        output_path = output_dir / "depth_all_colormaps.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()


def create_depth_visualization(raw_image, depth, output_dir=None, pred_only=False):
    """
    Create comprehensive depth visualization.

    Args:
        raw_image: Original BGR image
        depth: Raw depth map
        output_dir: Output directory for saving
        pred_only: If True, only save depth prediction
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "viz"
    output_dir.mkdir(exist_ok=True)

    # Apply official colormap
    depth_normalized, depth_colored = apply_colormap(depth, cmap_name="Spectral_r")

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Original Image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image", fontsize=14, weight="bold")
    ax1.axis("off")

    # 2. Grayscale Depth
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(depth_normalized, cmap="gray")
    ax2.set_title("Depth (Grayscale)", fontsize=14, weight="bold")
    ax2.axis("off")

    # 3. Colored Depth (Official)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(depth_normalized, cmap="Spectral_r")
    ax3.set_title("Depth (Spectral_r - Official)", fontsize=14, weight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # 4. Side-by-side comparison
    ax4 = fig.add_subplot(gs[1, :])
    side_by_side = create_side_by_side(raw_image, depth_colored)
    ax4.imshow(cv2.cvtColor(side_by_side, cv2.COLOR_BGR2RGB))
    ax4.set_title(
        "Side-by-Side Comparison (Official Style)", fontsize=14, weight="bold"
    )
    ax4.axis("off")

    # 5. Depth histogram
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(depth.flatten(), bins=100, color="steelblue", alpha=0.7, edgecolor="black")
    ax5.set_title("Depth Distribution", fontsize=14, weight="bold")
    ax5.set_xlabel("Depth Value", fontsize=12)
    ax5.set_ylabel("Frequency", fontsize=12)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(
        depth.mean(),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {depth.mean():.3f}",
    )
    ax5.axvline(
        np.median(depth),
        color="g",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(depth):.3f}",
    )
    ax5.legend(fontsize=10)

    # 6. Statistics
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")
    stats_text = f"""
Depth Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Minimum:    {depth.min():.6f}
Maximum:    {depth.max():.6f}
Mean:       {depth.mean():.6f}
Median:     {np.median(depth):.6f}
Std Dev:    {depth.std():.6f}
Range:      {depth.max() - depth.min():.6f}

Shape:      {depth.shape[0]} × {depth.shape[1]}
Pixels:     {depth.shape[0] * depth.shape[1]:,}
━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """
    ax6.text(
        0.1,
        0.5,
        stats_text,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # 7. Alternative colormap (plasma)
    ax7 = fig.add_subplot(gs[2, 2])
    im7 = ax7.imshow(depth_normalized, cmap="plasma")
    ax7.set_title("Depth (Plasma - Alternative)", fontsize=14, weight="bold")
    ax7.axis("off")
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

    plt.suptitle(
        "Depth Anything V2 - Complete Depth Visualization", fontsize=18, weight="bold"
    )

    # Save
    output_path = output_dir / "depth_complete_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved complete visualization: {output_path}")
    plt.show()

    # Save individual outputs
    if pred_only:
        cv2.imwrite(str(output_dir / "depth_pred_only.png"), depth_colored)
    else:
        cv2.imwrite(str(output_dir / "depth_grayscale.png"), depth_normalized)
        cv2.imwrite(str(output_dir / "depth_colored_spectral.png"), depth_colored)
        cv2.imwrite(str(output_dir / "depth_side_by_side.png"), side_by_side)
        print(f"Saved individual outputs to: {output_dir}")


def main():
    """Main application."""
    print("=" * 80)
    print("DEPTH VISUALIZATION APP")
    print("Based on Depth Anything V2 Official Implementation")
    print("=" * 80)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    script_dir = Path(__file__).parent.parent
    image_path = script_dir / "test_data" / "test_image.jpg"
    output_dir = script_dir / "viz"

    if not image_path.exists():
        print(f"\nError: Test image not found at {image_path}")
        print("Please ensure test_image.jpg exists in the test_data folder.")
        return

    # Load model
    model = load_depth_model(encoder="vits", device=device)
    if model is None:
        return

    # Infer depth
    print("\n" + "-" * 80)
    print("DEPTH INFERENCE")
    print("-" * 80)
    raw_image, depth = infer_depth(model, image_path)

    # Create visualizations
    print("\n" + "-" * 80)
    print("CREATING VISUALIZATIONS")
    print("-" * 80)

    # Complete visualization
    create_depth_visualization(raw_image, depth, output_dir)

    # Colormap comparison
    visualize_all_colormaps(depth, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nKey files:")
    print("  - depth_complete_visualization.png : Comprehensive view")
    print("  - depth_all_colormaps.png : Colormap comparison")
    print("  - depth_side_by_side.png : Official style comparison")
    print("  - depth_colored_spectral.png : Colored depth map")
    print("  - depth_grayscale.png : Grayscale depth map")


if __name__ == "__main__":
    main()
