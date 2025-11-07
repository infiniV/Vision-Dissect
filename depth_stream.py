import streamlit as st
import torch
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@st.cache_resource
def load_model():
    clear_gpu_memory()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")

    # Load model with explicit device map and lower memory usage
    model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if torch.cuda.is_available():
        model = model.to(device)
        clear_gpu_memory()

    return processor, model, device


def extract_intermediate_features(model, inputs, device):
    """Extract features from encoder layers and fusion stages."""
    activations = {}

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()

        return hook

    patch_encoder = model.depth_pro.encoder.patch_encoder.model
    handles = []

    if hasattr(patch_encoder, "encoder"):
        layers = patch_encoder.encoder.layer
        # Hook only first and last layers to save memory
        hook_indices = [0, len(layers) - 1]

        for idx in hook_indices:
            handle = layers[idx].register_forward_hook(
                get_activation(f"encoder_layer_{idx}")
            )
            handles.append(handle)

        # Hook only first fusion layer to save memory
        if len(model.fusion_stage.intermediate) > 0:
            handle = model.fusion_stage.intermediate[0].register_forward_hook(
                get_activation("fusion_0")
            )
            handles.append(handle)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda"):  # Use automatic mixed precision
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    activations["depth"] = outputs.predicted_depth.detach().cpu()

    # Clear GPU memory after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return activations, outputs


def visualize_features(activations):
    """Create visualization of intermediate features."""
    num_features = len(activations)
    fig, axes = plt.subplots(2, (num_features + 1) // 2, figsize=(16, 6))
    axes = axes.flatten()

    for idx, (name, feat) in enumerate(activations.items()):
        ax = axes[idx]

        if feat.dim() == 4:
            img = feat[0, 0].numpy()
        elif feat.dim() == 3:
            if feat.shape[1] > feat.shape[2]:
                B, N, C = feat.shape
                H = W = int(N**0.5)
                if H * W == N:
                    img = feat[0, :, 0].reshape(H, W).numpy()
                else:
                    approx_size = int(N**0.5)
                    subset = feat[0, : approx_size * approx_size, 0]
                    img = subset.reshape(approx_size, approx_size).numpy()
            else:
                img = feat[0].numpy()
        elif feat.dim() == 2:
            img = feat.numpy()
        else:
            ax.axis("off")
            continue

        ax.imshow(img, cmap="viridis")
        ax.set_title(name.replace("_", " ").title(), fontsize=9)
        ax.axis("off")

    for idx in range(len(activations), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    return fig


def create_depth_colormap_views(depth):
    """Create depth visualizations with different colormaps."""
    depth_np = depth.cpu().numpy() if torch.is_tensor(depth) else depth
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(depth_norm, cmap="viridis")
    axes[0].set_title("Viridis", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(depth_norm, cmap="turbo")
    axes[1].set_title("Turbo", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(depth_norm, cmap="plasma")
    axes[2].set_title("Plasma", fontsize=12)
    axes[2].axis("off")

    plt.tight_layout()
    return fig


def process_image_with_tracking(
    image, processor, model, device, progress_bar, status_text
):
    timings = {}

    # Step 1: Preprocessing
    status_text.text("Preprocessing image...")
    progress_bar.progress(0.1)
    start = time.time()
    inputs = processor(images=image, return_tensors="pt").to(device)
    timings["preprocessing"] = time.time() - start

    # Step 2: Extract intermediate features
    status_text.text("Running encoder and fusion layers...")
    progress_bar.progress(0.3)
    start = time.time()
    activations, outputs = extract_intermediate_features(model, inputs, device)
    timings["inference"] = time.time() - start

    # Step 3: Post-processing
    status_text.text("Post-processing depth map...")
    progress_bar.progress(0.9)
    start = time.time()
    post_processed = processor.post_process_depth_estimation(
        outputs, target_sizes=[(image.height, image.width)]
    )
    timings["postprocessing"] = time.time() - start

    progress_bar.progress(1.0)
    status_text.text("Complete!")

    depth = post_processed[0]["predicted_depth"]
    field_of_view = post_processed[0]["field_of_view"]
    focal_length = post_processed[0]["focal_length"]

    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
    depth_vis = (depth_normalized * 255).cpu().numpy().astype(np.uint8)

    timings["total"] = sum(timings.values())

    return depth_vis, depth, field_of_view, focal_length, timings, activations


def main():
    st.set_page_config(layout="wide")

    processor, model, device = load_model()

    image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if image_file:
        image = Image.open(image_file).convert("RGB")

        col1, col2 = st.columns(2)
        col1.image(image, use_container_width=True)

        if st.button("Process"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            depth_vis, depth, fov, focal, timings, activations = (
                process_image_with_tracking(
                    image, processor, model, device, progress_bar, status_text
                )
            )

            progress_bar.empty()
            status_text.empty()

            col2.image(depth_vis, use_container_width=True)

            st.write("=" * 80)
            st.write("DEPTH ESTIMATION RESULTS")
            st.write("=" * 80)
            st.write(f"Device: {device}")
            st.write(f"Image size: {image.size}")
            st.write(f"Depth map shape: {depth.shape}")
            st.write(f"Depth range: [{depth.min():.4f}, {depth.max():.4f}]")
            st.write(f"Field of view: {fov:.2f} degrees")
            st.write(f"Focal length: {focal:.2f} pixels")
            st.write("")
            st.write("TIMING BREAKDOWN")
            st.write(f"  Preprocessing: {timings['preprocessing']*1000:.2f} ms")
            st.write(f"  Inference: {timings['inference']*1000:.2f} ms")
            st.write(f"  Post-processing: {timings['postprocessing']*1000:.2f} ms")
            st.write(f"  Total: {timings['total']*1000:.2f} ms")
            st.write(f"  FPS: {1/timings['total']:.2f}")
            st.write("=" * 80)

            st.write("")
            st.write("INTERMEDIATE FEATURES")
            st.write("=" * 80)
            for name, feat in activations.items():
                st.write(f"  {name}: {feat.shape}")
            st.write("=" * 80)

            st.write("")
            st.write("DEPTH COLORMAPS")
            fig_colormaps = create_depth_colormap_views(depth)
            st.pyplot(fig_colormaps)

            st.write("")
            st.write("LAYER ACTIVATIONS")
            fig_features = visualize_features(activations)
            st.pyplot(fig_features)


if __name__ == "__main__":
    main()
