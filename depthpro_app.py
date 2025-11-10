import streamlit as st
import torch
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation


@st.cache_resource
def load_model():
    """Load model and processor with timing."""
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
    load_time = time.time() - start
    return model, image_processor, device, load_time


def process_depth(image, model, image_processor, device):
    """Run depth estimation with timing."""
    start = time.time()
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    inference_time = time.time() - start

    return (
        post_processed_output[0]["predicted_depth"],
        post_processed_output[0]["field_of_view"],
        post_processed_output[0]["focal_length"],
        inference_time,
    )


def normalize_depth(depth):
    """Normalize depth to 0-1 range."""
    depth_np = depth.detach().cpu().numpy()
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    return depth_norm


def create_depth_visualization(depth, colormap="turbo"):
    """Create depth visualization with colormap."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(depth, cmap=colormap)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def create_threshold_visualization(depth, threshold):
    """Create binary threshold visualization."""
    binary = (depth > threshold).astype(float)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(binary, cmap="gray")
    ax.axis("off")
    ax.set_title(f"Depth > {threshold:.2f}", fontsize=16)
    plt.tight_layout()
    return fig


def main():
    st.set_page_config(page_title="DepthPro Explorer", layout="wide")

    st.title("DepthPro Depth Estimation")

    # Sidebar
    st.sidebar.header("Configuration")

    # Load model
    with st.spinner("Loading model..."):
        model, image_processor, device, load_time = load_model()

    # Device info
    st.sidebar.subheader("System")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        st.sidebar.metric("Device", "GPU")
        st.sidebar.caption(gpu_name)
    else:
        st.sidebar.metric("Device", "CPU")

    st.sidebar.metric("Model Load Time", f"{load_time:.2f}s")

    # Image upload
    st.sidebar.subheader("Input")
    uploaded_file = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert("RGB")

        # Process
        with st.spinner("Processing..."):
            depth, fov, focal_length, inference_time = process_depth(
                image, model, image_processor, device
            )

        depth_normalized = normalize_depth(depth)
        depth_np = depth.detach().cpu().numpy()

        # Metrics
        st.sidebar.subheader("Performance")
        st.sidebar.metric("Inference Time", f"{inference_time:.3f}s")

        st.sidebar.subheader("Output Metadata")
        st.sidebar.metric("Field of View", f"{fov:.2f}°")
        st.sidebar.metric("Focal Length", f"{focal_length:.2f}px")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Depth Statistics")
            stats_col1, stats_col2, stats_col3 = st.columns(3)
            stats_col1.metric("Min Depth", f"{depth_np.min():.2f}m")
            stats_col2.metric("Max Depth", f"{depth_np.max():.2f}m")
            stats_col3.metric("Mean Depth", f"{depth_np.mean():.2f}m")

        with col2:
            st.subheader("Image Info")
            info_col1, info_col2, info_col3 = st.columns(3)
            info_col1.metric("Width", f"{image.width}px")
            info_col2.metric("Height", f"{image.height}px")
            info_col3.metric("Output Shape", f"{depth.shape[0]}×{depth.shape[1]}")

        # Visualization
        st.divider()

        tab1, tab2, tab3 = st.tabs(["Original", "Depth Map", "Threshold"])

        with tab1:
            st.image(image, use_container_width=True)

        with tab2:
            fig = create_depth_visualization(depth_normalized, "turbo")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            threshold = st.slider(
                "Depth Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01,
                help="Objects with normalized depth above this threshold will be shown in white",
            )

            fig = create_threshold_visualization(depth_normalized, threshold)
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Upload an image to begin depth estimation")


if __name__ == "__main__":
    main()
