import torch
import matplotlib.pyplot as plt
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
from PIL import Image


def inspect_model_architecture(model):
    """Print detailed model architecture."""
    print("\n" + "=" * 80)
    print("DEPTHPRO MODEL ARCHITECTURE")
    print("=" * 80)

    # Main model components
    print("\nMain Components:")
    print(f"  depth_pro: {type(model.depth_pro).__name__}")
    print(f"  fusion_stage: {type(model.fusion_stage).__name__}")
    print(f"  head: {type(model.head).__name__}")
    if hasattr(model, "fov_model") and model.fov_model is not None:
        print(f"  fov_model: {type(model.fov_model).__name__}")

    # Encoder details
    encoder = model.depth_pro.encoder
    print(f"\nEncoder Structure:")
    print(f"  patch_encoder: {type(encoder.patch_encoder).__name__}")
    print(f"  image_encoder: {type(encoder.image_encoder).__name__}")

    # Patch encoder (Dinov2)
    if hasattr(encoder.patch_encoder, "model"):
        patch_backbone = encoder.patch_encoder.model
        print(f"\n  Patch Encoder (Dinov2):")
        print(f"    Embeddings: {type(patch_backbone.embeddings).__name__}")
        print(f"    Num layers: {len(patch_backbone.encoder.layer)}")
        print(f"    Layer type: {type(patch_backbone.encoder.layer[0]).__name__}")
        if hasattr(patch_backbone, "layernorm"):
            print(f"    LayerNorm: {type(patch_backbone.layernorm).__name__}")

    # Image encoder (Dinov2)
    if hasattr(encoder.image_encoder, "model"):
        image_backbone = encoder.image_encoder.model
        print(f"\n  Image Encoder (Dinov2):")
        print(f"    Embeddings: {type(image_backbone.embeddings).__name__}")
        print(f"    Num layers: {len(image_backbone.encoder.layer)}")
        print(f"    Layer type: {type(image_backbone.encoder.layer[0]).__name__}")
        if hasattr(image_backbone, "layernorm"):
            print(f"    LayerNorm: {type(image_backbone.layernorm).__name__}")

    # Neck
    neck = model.depth_pro.neck
    print(f"\nNeck Structure:")
    print(f"  feature_upsample: {type(neck.feature_upsample).__name__}")
    print(f"  feature_projection: {type(neck.feature_projection).__name__}")

    # Fusion stage
    fusion = model.fusion_stage
    print(f"\nFeature Fusion Stage:")
    print(f"  Intermediate layers: {len(fusion.intermediate)}")
    print(f"  Final layer: {type(fusion.final).__name__}")

    # Head
    print(f"\nDepth Estimation Head:")
    if hasattr(model.head, "layers"):
        for i, layer in enumerate(model.head.layers):
            print(f"  Layer {i}: {type(layer).__name__}")
            if hasattr(layer, "in_channels") and hasattr(layer, "out_channels"):
                print(f"           {layer.in_channels} -> {layer.out_channels}")

    print("\n" + "=" * 80)


def extract_intermediate_features(model, image_processor, image, device):
    """Extract features from different stages of the model."""
    print("\nExtracting intermediate features...")

    activations = {}

    # Hook for patch encoder layers
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()

        return hook

    # Register hooks on patch encoder layers
    patch_encoder = model.depth_pro.encoder.patch_encoder.model
    if hasattr(patch_encoder, "encoder"):
        layers = patch_encoder.encoder.layer
        # Hook early, middle, and late layers
        hook_indices = [0, len(layers) // 2, len(layers) - 1]
        handles = []

        for idx in hook_indices:
            handle = layers[idx].register_forward_hook(
                get_activation(f"patch_encoder_layer_{idx}")
            )
            handles.append(handle)

        # Hook fusion blocks
        for i, layer in enumerate(model.fusion_stage.intermediate):
            handle = layer.register_forward_hook(get_activation(f"fusion_layer_{i}"))
            handles.append(handle)

    # Run inference
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Add final output
    activations["final_depth"] = outputs.predicted_depth.detach().cpu()

    print(f"  Captured {len(activations)} feature maps")
    for name, feat in activations.items():
        print(f"    {name}: {feat.shape}")

    return activations, outputs


def visualize_layer_features(activations, save_path="../viz/depthpro_features.png"):
    """Visualize features from different layers."""
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    num_features = len(activations)
    fig, axes = plt.subplots(2, (num_features + 1) // 2, figsize=(20, 8))
    axes = axes.flatten()

    for idx, (name, feat) in enumerate(activations.items()):
        ax = axes[idx]

        # Handle different tensor shapes
        if feat.dim() == 4:  # [B, C, H, W]
            # Show first channel
            img = feat[0, 0].numpy()
        elif feat.dim() == 3:  # [B, H, W] or [B, N, C]
            if feat.shape[1] > feat.shape[2]:  # Sequence format [B, N, C]
                # Reshape to spatial if possible
                B, N, C = feat.shape
                H = W = int(N**0.5)
                if H * W == N:
                    img = feat[0, :, 0].reshape(H, W).numpy()
                else:
                    img = feat[0, :100, 0].numpy().reshape(10, 10)  # Approximate
            else:  # Spatial format [B, H, W]
                img = feat[0].numpy()
        elif feat.dim() == 2:  # [H, W]
            img = feat.numpy()
        else:
            ax.text(0.5, 0.5, f"Shape: {feat.shape}", ha="center", va="center")
            ax.axis("off")
            continue

        im = ax.imshow(img, cmap="viridis")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(len(activations), len(axes)):
        axes[idx].axis("off")

    plt.suptitle("DepthPro Intermediate Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {save_path}")


def compare_encoder_layers(model, image_processor, image, device, num_layers=4):
    """Compare features from different encoder layers."""
    print(f"\nComparing {num_layers} encoder layers...")

    patch_encoder = model.depth_pro.encoder.patch_encoder.model
    if not hasattr(patch_encoder, "encoder"):
        print("  Cannot access encoder layers")
        return

    layers = patch_encoder.encoder.layer
    total_layers = len(layers)

    # Select evenly spaced layers
    layer_indices = [int(i * total_layers / num_layers) for i in range(num_layers)]

    activations = {}
    handles = []

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()

        return hook

    for idx in layer_indices:
        handle = layers[idx].register_forward_hook(get_activation(f"layer_{idx}"))
        handles.append(handle)

    # Run inference
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**inputs)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Visualize
    fig, axes = plt.subplots(1, num_layers, figsize=(20, 5))

    for idx, (name, feat) in enumerate(activations.items()):
        ax = axes[idx]

        # Extract spatial features [B, N, C] -> [H, W]
        if feat.dim() == 3:
            B, N, C = feat.shape
            H = W = int(N**0.5)
            if H * W == N:
                # Average across channels and reshape
                img = feat[0, :, :].mean(dim=1).reshape(H, W).numpy()
            else:
                # Cannot reshape perfectly, take subset
                approx_size = int(N**0.5)
                subset = feat[0, : approx_size * approx_size, :].mean(dim=1)
                img = subset.reshape(approx_size, approx_size).numpy()
        else:
            img = feat[0, 0].numpy() if feat.dim() == 4 else feat[0].numpy()

        im = ax.imshow(img, cmap="viridis")
        layer_num = int(name.split("_")[-1])
        ax.set_title(f"Layer {layer_num}/{total_layers-1}", fontsize=12, weight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("DepthPro Patch Encoder Layer Progression", fontsize=16)
    plt.tight_layout()
    plt.savefig("../viz/depthpro_layer_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: viz/depthpro_layer_comparison.png")


def visualize_attention_patterns(model, image_processor, image, device, layer_idx=11):
    """Extract and visualize attention patterns from a specific layer."""
    print(f"\nExtracting attention from layer {layer_idx}...")

    patch_encoder = model.depth_pro.encoder.patch_encoder.model
    if not hasattr(patch_encoder, "encoder"):
        print("  Cannot access encoder layers")
        return

    layers = patch_encoder.encoder.layer
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1

    # Try to enable eager attention for attention weights
    try:
        model.set_attn_implementation("eager")
        print("  Set attention implementation to eager mode")
    except:
        print("  Cannot enable eager attention, skipping attention visualization")
        return

    attention_weights = {}

    def get_attention(name):
        def hook(module, input, output):
            # Dinov2 attention output is a tuple: (hidden_states, attention_weights)
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights[name] = output[1].detach().cpu()

        return hook

    # Hook the attention module
    handle = layers[layer_idx].attention.register_forward_hook(get_attention("attn"))

    # Run inference with output_attentions=True
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    handle.remove()

    if not attention_weights:
        print("  No attention weights captured")
        return

    # Visualize attention patterns
    attn = attention_weights["attn"]  # [B, num_heads, seq_len, seq_len]
    if attn is None:
        print("  Attention is None")
        return

    print(f"  Attention shape: {attn.shape}")

    num_heads = min(8, attn.shape[1])
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        ax = axes[head_idx]

        # Get attention for this head [seq_len, seq_len]
        head_attn = attn[0, head_idx].numpy()

        # Show attention from CLS token (first token) to all patches
        im = ax.imshow(head_attn[:50, :50], cmap="hot", interpolation="nearest")
        ax.set_title(f"Head {head_idx}", fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"DepthPro Attention Patterns (Layer {layer_idx})", fontsize=16)
    plt.tight_layout()
    plt.savefig("../viz/depthpro_attention.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: viz/depthpro_attention.png")


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading DepthPro model...")
    image_processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
    model = DepthProForDepthEstimation.from_pretrained(
        "apple/DepthPro-hf", use_fov_model=True
    ).to(device)
    print("Model loaded")

    # Inspect architecture
    inspect_model_architecture(model)

    # Load image
    print("\nLoading test image...")
    image = Image.open("test_data/test_image.jpg")
    print(f"  Image size: {image.size}")

    # Extract and visualize features
    activations, outputs = extract_intermediate_features(
        model, image_processor, image, device
    )
    visualize_layer_features(activations)

    # Compare encoder layers
    compare_encoder_layers(model, image_processor, image, device, num_layers=4)

    # Note: Attention visualization requires eager mode which may not be available
    # visualize_attention_patterns(model, image_processor, image, device, layer_idx=11)

    print("\nDissection complete!")


if __name__ == "__main__":
    main()
