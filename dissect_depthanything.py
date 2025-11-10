import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np


def inspect_model_architecture(model):
    """Print detailed model architecture."""
    print("\n" + "=" * 80)
    print("DEPTH ANYTHING V2 MODEL ARCHITECTURE")
    print("=" * 80)

    print("\nMain Components:")
    print(f"  encoder: {model.encoder}")
    print(f"  pretrained: {type(model.pretrained).__name__}")
    print(f"  depth_head: {type(model.depth_head).__name__}")

    print(f"\nIntermediate Layer Indices:")
    for enc, indices in model.intermediate_layer_idx.items():
        print(f"  {enc}: {indices}")

    pretrained = model.pretrained
    print(f"\nDINOv2 Encoder Structure:")
    print(f"  patch_embed: {type(pretrained.patch_embed).__name__}")
    print(f"  embed_dim: {pretrained.embed_dim}")
    print(f"  num_heads: {pretrained.num_heads}")
    print(f"  depth: {len(pretrained.blocks)}")
    print(f"  mlp_ratio: {pretrained.mlp_ratio}")
    
    if hasattr(pretrained, 'patch_embed'):
        pe = pretrained.patch_embed
        print(f"\n  Patch Embedding:")
        print(f"    img_size: {pe.img_size}")
        print(f"    patch_size: {pe.patch_size}")
        print(f"    embed_dim: {pe.embed_dim}")
        if hasattr(pe, 'proj'):
            print(f"    projection: {type(pe.proj).__name__}")

    print(f"\n  Transformer Blocks ({len(pretrained.blocks)} blocks):")
    if len(pretrained.blocks) > 0:
        block = pretrained.blocks[0]
        print(f"    Block type: {type(block).__name__}")
        if hasattr(block, 'attn'):
            print(f"    Attention: {type(block.attn).__name__}")
        if hasattr(block, 'mlp'):
            print(f"    MLP: {type(block.mlp).__name__}")
        if hasattr(block, 'norm1'):
            print(f"    Norm1: {type(block.norm1).__name__}")
        if hasattr(block, 'norm2'):
            print(f"    Norm2: {type(block.norm2).__name__}")

    print(f"\n  Final Norm: {type(pretrained.norm).__name__}")

    depth_head = model.depth_head
    print(f"\nDPT Head Structure:")
    print(f"  use_clstoken: {depth_head.use_clstoken}")
    print(f"  projects: {len(depth_head.projects)} projection layers")
    print(f"  resize_layers: {len(depth_head.resize_layers)} resize layers")
    
    for i, proj in enumerate(depth_head.projects):
        print(f"    Project {i}: {proj.in_channels} -> {proj.out_channels}")
    
    print(f"\n  Scratch Modules:")
    if hasattr(depth_head.scratch, 'layer1_rn'):
        print(f"    layer1_rn: {type(depth_head.scratch.layer1_rn).__name__}")
        print(f"    layer2_rn: {type(depth_head.scratch.layer2_rn).__name__}")
        print(f"    layer3_rn: {type(depth_head.scratch.layer3_rn).__name__}")
        print(f"    layer4_rn: {type(depth_head.scratch.layer4_rn).__name__}")
    
    if hasattr(depth_head.scratch, 'refinenet1'):
        print(f"    refinenet1: {type(depth_head.scratch.refinenet1).__name__}")
        print(f"    refinenet2: {type(depth_head.scratch.refinenet2).__name__}")
        print(f"    refinenet3: {type(depth_head.scratch.refinenet3).__name__}")
        print(f"    refinenet4: {type(depth_head.scratch.refinenet4).__name__}")
    
    if hasattr(depth_head.scratch, 'output_conv1'):
        print(f"    output_conv1: {type(depth_head.scratch.output_conv1).__name__}")
    if hasattr(depth_head.scratch, 'output_conv2'):
        print(f"    output_conv2: {type(depth_head.scratch.output_conv2).__name__}")

    print("\n" + "=" * 80)


def extract_intermediate_features(model, image, device):
    """Extract features from different stages of the model."""
    print("\nExtracting intermediate features...")
    
    activations = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu()
            else:
                activations[name] = output.detach().cpu()
        return hook
    
    pretrained = model.pretrained
    blocks = pretrained.blocks
    
    layer_indices = model.intermediate_layer_idx[model.encoder]
    handles = []
    
    for idx in layer_indices:
        if idx < len(blocks):
            handle = blocks[idx].register_forward_hook(get_activation(f"block_{idx}"))
            handles.append(handle)
    
    if hasattr(model.depth_head.scratch, 'refinenet1'):
        handle = model.depth_head.scratch.refinenet1.register_forward_hook(
            get_activation("refinenet1"))
        handles.append(handle)
        handle = model.depth_head.scratch.refinenet2.register_forward_hook(
            get_activation("refinenet2"))
        handles.append(handle)
        handle = model.depth_head.scratch.refinenet3.register_forward_hook(
            get_activation("refinenet3"))
        handles.append(handle)
        handle = model.depth_head.scratch.refinenet4.register_forward_hook(
            get_activation("refinenet4"))
        handles.append(handle)
    
    with torch.no_grad():
        if image.shape[-1] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image = image.to(device)
        
        depth = model(image)
    
    for handle in handles:
        handle.remove()
    
    activations["final_depth"] = depth.detach().cpu()
    
    print(f"  Captured {len(activations)} feature maps")
    for name, feat in activations.items():
        print(f"    {name}: {feat.shape}")
    
    return activations


def visualize_layer_features(activations, save_path="viz/depthanything_features.png"):
    """Visualize features from different layers."""
    num_features = len(activations)
    rows = 2
    cols = (num_features + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
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
                    img = feat[0, :100].mean(dim=1).numpy().reshape(10, 10)
            else:
                img = feat[0].numpy()
        elif feat.dim() == 2:
            img = feat.numpy()
        else:
            ax.text(0.5, 0.5, f"Shape: {feat.shape}", ha="center", va="center")
            ax.axis("off")
            continue
        
        im = ax.imshow(img, cmap="viridis")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for idx in range(len(activations), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Depth Anything V2 Intermediate Features", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved: {save_path}")


def compare_encoder_layers(model, image, device, num_samples=4):
    """Compare features from different encoder layers."""
    print(f"\nComparing {num_samples} encoder layers...")
    
    pretrained = model.pretrained
    blocks = pretrained.blocks
    total_blocks = len(blocks)
    
    layer_indices = [int(i * total_blocks / num_samples) for i in range(num_samples)]
    
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
        if idx < total_blocks:
            handle = blocks[idx].register_forward_hook(get_activation(f"block_{idx}"))
            handles.append(handle)
    
    with torch.no_grad():
        if image.shape[-1] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image = image.to(device)
        
        _ = model(image)
    
    for handle in handles:
        handle.remove()
    
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    
    for idx, (name, feat) in enumerate(activations.items()):
        ax = axes[idx]
        
        if feat.dim() == 3:
            B, N, C = feat.shape
            H = W = int(N**0.5)
            if H * W == N:
                img = feat[0, :, :].mean(dim=1).reshape(H, W).numpy()
            else:
                approx_size = int(N**0.5)
                subset = feat[0, :approx_size * approx_size, :].mean(dim=1)
                img = subset.reshape(approx_size, approx_size).numpy()
        else:
            img = feat[0, 0].numpy() if feat.dim() == 4 else feat[0].numpy()
        
        im = ax.imshow(img, cmap="viridis")
        layer_num = int(name.split("_")[-1])
        ax.set_title(f"Block {layer_num}/{total_blocks-1}", fontsize=12, weight="bold")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle("Depth Anything V2 Encoder Block Progression", fontsize=16)
    plt.tight_layout()
    plt.savefig("viz/depthanything_layer_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: viz/depthanything_layer_comparison.png")


def visualize_dpt_head_stages(model, image, device):
    """Visualize DPT head refinement stages."""
    print("\nVisualizing DPT head refinement stages...")
    
    activations = {}
    handles = []
    
    def get_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().cpu()
        return hook
    
    if hasattr(model.depth_head.scratch, 'layer1_rn'):
        handles.append(model.depth_head.scratch.layer1_rn.register_forward_hook(
            get_activation("layer1_rn")))
        handles.append(model.depth_head.scratch.layer2_rn.register_forward_hook(
            get_activation("layer2_rn")))
        handles.append(model.depth_head.scratch.layer3_rn.register_forward_hook(
            get_activation("layer3_rn")))
        handles.append(model.depth_head.scratch.layer4_rn.register_forward_hook(
            get_activation("layer4_rn")))
    
    if hasattr(model.depth_head.scratch, 'refinenet1'):
        handles.append(model.depth_head.scratch.refinenet1.register_forward_hook(
            get_activation("refinenet1")))
        handles.append(model.depth_head.scratch.refinenet2.register_forward_hook(
            get_activation("refinenet2")))
        handles.append(model.depth_head.scratch.refinenet3.register_forward_hook(
            get_activation("refinenet3")))
        handles.append(model.depth_head.scratch.refinenet4.register_forward_hook(
            get_activation("refinenet4")))
    
    with torch.no_grad():
        if image.shape[-1] == 3:
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
        else:
            image = image.to(device)
        
        depth = model(image)
    
    for handle in handles:
        handle.remove()
    
    activations["final_output"] = depth.detach().cpu()
    
    num_activations = len(activations)
    rows = 2
    cols = (num_activations + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (name, feat) in enumerate(activations.items()):
        ax = axes[idx]
        
        if feat.dim() == 4:
            img = feat[0, 0].numpy()
        elif feat.dim() == 3:
            img = feat[0].numpy()
        elif feat.dim() == 2:
            img = feat.numpy()
        else:
            ax.axis("off")
            continue
        
        im = ax.imshow(img, cmap="plasma")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    for idx in range(len(activations), len(axes)):
        axes[idx].axis("off")
    
    plt.suptitle("Depth Anything V2 DPT Head Refinement Stages", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig("viz/depthanything_dpt_stages.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: viz/depthanything_dpt_stages.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading Depth Anything V2 model...")
    
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("Error: depth_anything_v2 module not found.")
        print("Please install it or ensure it's in your Python path.")
        return
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    encoder = 'vits'
    
    model = DepthAnythingV2(**model_configs[encoder])
    model = model.to(device)
    model.eval()
    print("Model loaded")
    
    inspect_model_architecture(model)
    
    print("\nLoading test image...")
    image = cv2.imread("test_image.jpg")
    if image is None:
        print("Error: test_image.jpg not found")
        return
    
    print(f"  Image shape: {image.shape}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0).float()
    
    activations = extract_intermediate_features(model, image_tensor, device)
    visualize_layer_features(activations)
    
    compare_encoder_layers(model, image_tensor, device, num_samples=4)
    
    visualize_dpt_head_stages(model, image_tensor, device)
    
    print("\nDissection complete!")


if __name__ == "__main__":
    main()
