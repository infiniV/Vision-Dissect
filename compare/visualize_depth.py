"""
Depth Estimation Visualization
Properly visualize depth maps with colormaps following Depth Anything V2 conventions
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


def visualize_depth_comparison(image_path, output_dir=None):
    """
    Visualize depth estimation with proper colormap normalization.
    Based on Depth Anything V2 official visualization.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "viz"
    output_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load original image
    print(f"\nLoading image: {image_path}")
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image shape: {raw_image.shape}")

    # Try to load Depth Anything V2 model
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

        encoder = "vits"  # Use small model for faster inference
        print(f"\nLoading Depth Anything V2 ({encoder})...")

        model = DepthAnythingV2(**model_configs[encoder])
        model = model.to(device).eval()

        # Infer depth
        print("Inferring depth...")
        with torch.no_grad():
            depth = model.infer_image(raw_image, input_size=518)

        print(f"Raw depth range: [{depth.min():.4f}, {depth.max():.4f}]")

        # Create visualization following official Depth Anything V2 approach
        # Use Spectral_r colormap (reversed Spectral)
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")

        # Normalize depth to 0-255
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)

        # Apply colormap
        depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)
        depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)

        # Create side-by-side comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        axes[0].imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Original Image", fontsize=14, weight="bold")
        axes[0].axis("off")

        # Grayscale depth
        axes[1].imshow(depth_normalized, cmap="gray")
        axes[1].set_title("Depth Map (Grayscale)", fontsize=14, weight="bold")
        axes[1].axis("off")

        # Colored depth with colorbar
        im = axes[2].imshow(depth_normalized, cmap="Spectral_r")
        axes[2].set_title("Depth Map (Colored)", fontsize=14, weight="bold")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

        plt.suptitle("Depth Anything V2 - Depth Estimation", fontsize=16, weight="bold")
        plt.tight_layout()

        output_path = output_dir / "depth_estimation_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved comparison: {output_path}")
        plt.show()

        # Save individual outputs
        cv2.imwrite(str(output_dir / "depth_grayscale.png"), depth_normalized)
        cv2.imwrite(str(output_dir / "depth_colored.png"), depth_colored_bgr)

        # Save side-by-side comparison (like official implementation)
        margin_width = 50
        split_region = (
            np.ones((raw_image.shape[0], margin_width, 3), dtype=np.uint8) * 255
        )
        combined = cv2.hconcat([raw_image, split_region, depth_colored_bgr])
        cv2.imwrite(str(output_dir / "depth_side_by_side.png"), combined)
        print(f"Saved side-by-side: {output_dir / 'depth_side_by_side.png'}")

        # Create depth visualization with different colormaps
        visualize_colormaps(depth_normalized, output_dir)

    except ImportError:
        print("\nWarning: depth_anything_v2 module not found.")
        print("Please clone the Depth-Anything-V2 repository:")
        print("  git clone https://github.com/DepthAnything/Depth-Anything-V2")
        print("And copy the depth_anything_v2 folder to this directory.")


def visualize_colormaps(depth_normalized, output_dir):
    """Compare different colormaps for depth visualization."""
    print("\nGenerating colormap comparisons...")

    colormaps = ["Spectral_r", "viridis", "plasma", "magma", "inferno", "turbo"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, cmap_name in enumerate(colormaps):
        cmap = matplotlib.colormaps.get_cmap(cmap_name)
        im = axes[idx].imshow(depth_normalized, cmap=cmap_name)
        axes[idx].set_title(f"{cmap_name}", fontsize=14, weight="bold")
        axes[idx].axis("off")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Depth Map Visualization - Different Colormaps", fontsize=16, weight="bold"
    )
    plt.tight_layout()

    output_path = output_dir / "depth_colormaps_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved colormap comparison: {output_path}")
    plt.show()


def visualize_depth_details(image_path, output_dir=None):
    """
    Visualize depth with detailed analysis including histogram and slices.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "viz"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load image
    raw_image = cv2.imread(str(image_path))
    if raw_image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    try:
        from depth_anything_v2.dpt import DepthAnythingV2

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            },
        }

        encoder = "vits"
        print(f"\nLoading Depth Anything V2 for detailed analysis...")

        model = DepthAnythingV2(**model_configs[encoder])
        model = model.to(device).eval()

        # Infer depth
        with torch.no_grad():
            depth = model.infer_image(raw_image, input_size=518)

        # Create detailed visualization
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Original image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image", fontsize=12, weight="bold")
        ax1.axis("off")

        # Depth map (Spectral_r)
        ax2 = fig.add_subplot(gs[0, 1])
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        im2 = ax2.imshow(depth_normalized, cmap="Spectral_r")
        ax2.set_title("Depth Map (Spectral_r)", fontsize=12, weight="bold")
        ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Depth histogram
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.hist(depth.flatten(), bins=50, color="steelblue", alpha=0.7)
        ax3.set_title("Depth Distribution", fontsize=12, weight="bold")
        ax3.set_xlabel("Depth Value")
        ax3.set_ylabel("Frequency")
        ax3.grid(True, alpha=0.3)

        # Horizontal slice
        ax4 = fig.add_subplot(gs[1, :])
        mid_row = depth.shape[0] // 2
        ax4.plot(depth[mid_row, :], color="darkblue", linewidth=2)
        ax4.set_title(
            f"Horizontal Depth Slice (Row {mid_row})", fontsize=12, weight="bold"
        )
        ax4.set_xlabel("Column")
        ax4.set_ylabel("Depth")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(
            y=depth.mean(), color="r", linestyle="--", label=f"Mean: {depth.mean():.2f}"
        )
        ax4.legend()

        # Vertical slice
        ax5 = fig.add_subplot(gs[2, 0])
        mid_col = depth.shape[1] // 2
        ax5.plot(
            depth[:, mid_col],
            np.arange(len(depth[:, mid_col])),
            color="darkgreen",
            linewidth=2,
        )
        ax5.set_title(f"Vertical Slice (Col {mid_col})", fontsize=12, weight="bold")
        ax5.set_xlabel("Depth")
        ax5.set_ylabel("Row")
        ax5.grid(True, alpha=0.3)
        ax5.invert_yaxis()

        # Statistics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis("off")
        stats_text = f"""
Depth Statistics:
━━━━━━━━━━━━━━━━━━
Min:    {depth.min():.4f}
Max:    {depth.max():.4f}
Mean:   {depth.mean():.4f}
Median: {np.median(depth):.4f}
Std:    {depth.std():.4f}

Shape:  {depth.shape}
━━━━━━━━━━━━━━━━━━
        """
        ax6.text(
            0.1,
            0.5,
            stats_text,
            fontsize=11,
            family="monospace",
            verticalalignment="center",
        )

        # 3D-like visualization
        ax7 = fig.add_subplot(gs[2, 2], projection="3d")
        # Downsample for 3D plot
        step = 10
        y_grid = np.arange(0, depth.shape[0], step)
        x_grid = np.arange(0, depth.shape[1], step)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = depth[::step, ::step]

        surf = ax7.plot_surface(X, Y, Z, cmap="Spectral_r", alpha=0.8)
        ax7.set_title("3D Depth Surface", fontsize=12, weight="bold")
        ax7.set_xlabel("X")
        ax7.set_ylabel("Y")
        ax7.set_zlabel("Depth")

        plt.suptitle("Depth Estimation - Detailed Analysis", fontsize=16, weight="bold")

        output_path = output_dir / "depth_detailed_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved detailed analysis: {output_path}")
        plt.show()

    except ImportError:
        print("\nWarning: depth_anything_v2 module not found.")


def main():
    """Main function to run all visualizations."""
    script_dir = Path(__file__).parent.parent
    image_path = script_dir / "test_data" / "test_image.jpg"
    output_dir = script_dir / "viz"

    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}")
        return

    print("=" * 80)
    print("DEPTH VISUALIZATION TOOL")
    print("=" * 80)

    # Run visualizations
    visualize_depth_comparison(image_path, output_dir)
    visualize_depth_details(image_path, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
