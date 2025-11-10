import torch
import matplotlib.pyplot as plt
from ultralytics import SAM
from PIL import Image
import numpy as np


def load_sam(model_path):
    return SAM(model_path)


def print_model_structure(model):
    print("\nModel Structure:")
    print("=" * 60)
    for name, module in model.model.named_children():
        print(f"\n{name}: {type(module).__name__}")

        param_count = sum(p.numel() for p in module.parameters())
        print(f"  Parameters: {param_count:,}")

        if hasattr(module, "named_children"):
            for sub_name, sub_module in module.named_children():
                print(f"  - {sub_name}: {type(sub_module).__name__}")


def run_inference(model, image_path):
    results = model(image_path, verbose=False)[0]
    return results


def extract_hidden_layer(model, image_path, component="image_encoder"):
    activation = {}

    def hook(module, input, output):
        if torch.is_tensor(output):
            activation["data"] = output.detach().cpu()
        elif isinstance(output, (tuple, list)):
            activation["data"] = [
                o.detach().cpu() if torch.is_tensor(o) else o for o in output
            ]

    if component == "image_encoder":
        target = model.model.image_encoder
    elif component == "mask_decoder":
        target = model.model.mask_decoder
    elif component == "prompt_encoder":
        target = model.model.prompt_encoder
    else:
        return None

    handle = target.register_forward_hook(hook)
    _ = model(image_path, verbose=False)
    handle.remove()

    return activation.get("data")


def inspect_results(results):
    print("\nResults:")
    print("=" * 60)
    print(f"Type: {type(results)}")

    print(f"\nAvailable attributes:")
    attrs = [attr for attr in dir(results) if not attr.startswith("_")]
    for attr in attrs[:20]:
        print(f"  {attr}")

    if hasattr(results, "masks") and results.masks is not None:
        print(f"\nMasks:")
        print(f"  Shape: {results.masks.data.shape}")
        print(
            f"  Range: [{results.masks.data.min():.3f}, {results.masks.data.max():.3f}]"
        )
        print(f"  Dtype: {results.masks.data.dtype}")
        print(f"  Unique values: {torch.unique(results.masks.data).numel()}")

    if hasattr(results, "boxes") and results.boxes is not None:
        print(f"\nBoxes: {len(results.boxes)}")
        if len(results.boxes) > 0:
            print(f"  Data shape: {results.boxes.data.shape}")
            print(f"  Sample box: {results.boxes.data[0].tolist()}")
            if hasattr(results.boxes, "conf"):
                print(
                    f"  Confidence range: [{results.boxes.conf.min():.3f}, {results.boxes.conf.max():.3f}]"
                )

    if hasattr(results, "orig_shape"):
        print(f"\nOriginal image shape: {results.orig_shape}")


def visualize_result(results, title="SAM Output"):
    plotted = results.plot()

    plt.figure(figsize=(12, 8))
    plt.imshow(plotted[..., ::-1])
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    model = load_sam("../models/mobile_sam.pt")

    print_model_structure(model)

    results = run_inference(model, "../test_data/test_image.jpg")

    inspect_results(results)

    print("\n" + "=" * 60)
    print("Extracting Hidden Layers")
    print("=" * 60)

    encoder_output = None
    decoder_output = None

    for component in ["image_encoder", "prompt_encoder", "mask_decoder"]:
        print(f"\n{component}:")
        hidden = extract_hidden_layer(model, "../test_data/test_image.jpg", component)

        if component == "image_encoder":
            encoder_output = hidden
        elif component == "mask_decoder":
            decoder_output = hidden

        if hidden is not None:
            if isinstance(hidden, list):
                print(f"  Outputs: {len(hidden)} tensors")
                for i, tensor in enumerate(hidden):
                    if torch.is_tensor(tensor):
                        print(
                            f"    [{i}] shape={tensor.shape} range=[{tensor.min():.3f}, {tensor.max():.3f}]"
                        )
            elif torch.is_tensor(hidden):
                print(f"  Shape: {hidden.shape}")
                print(f"  Range: [{hidden.min():.3f}, {hidden.max():.3f}]")
                print(f"  Dtype: {hidden.dtype}")

    print("\n" + "=" * 60)
    print("Visualizing Hidden Layers")
    print("=" * 60)

    # Visualize encoder output (like YOLO layer 22)
    if encoder_output is not None and torch.is_tensor(encoder_output):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8))
        fig.suptitle("SAM Image Encoder Output (8 channels)", fontsize=14)

        for idx in range(min(8, encoder_output.shape[1])):
            ax = axes[idx // 4, idx % 4]
            feature_map = encoder_output[0, idx].detach().cpu().numpy()
            im = ax.imshow(feature_map, cmap="viridis")
            ax.set_title(f"Ch {idx}")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        plt.savefig("../viz/sam_encoder.png", dpi=150, bbox_inches="tight")
        print("Saved: viz/sam_encoder.png")
        plt.show()

    # Visualize decoder output masks
    if decoder_output is not None and isinstance(decoder_output, list):
        masks_tensor = decoder_output[0]
        if torch.is_tensor(masks_tensor) and len(masks_tensor.shape) == 4:
            fig, axes = plt.subplots(2, 4, figsize=(15, 8))
            fig.suptitle(
                f"SAM Mask Decoder Output (8/{masks_tensor.shape[0]} masks)",
                fontsize=14,
            )

            for idx in range(min(8, masks_tensor.shape[0])):
                ax = axes[idx // 4, idx % 4]
                mask = masks_tensor[idx, 0].detach().cpu().numpy()
                im = ax.imshow(mask, cmap="gray")
                ax.set_title(f"Mask {idx}")
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046)

            plt.tight_layout()
            plt.savefig("../viz/sam_decoder.png", dpi=150, bbox_inches="tight")
            print("Saved: viz/sam_decoder.png")
            plt.show()

    visualize_result(results, "MobileSAM")
