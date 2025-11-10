"""
Unified Vision Model Benchmark System

Sequentially evaluates vision models with comprehensive layer dissection,
memory-efficient processing, and detailed performance metrics.
"""

import sys
import time
import random
import gc
import json
import csv
import traceback
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# Constants
NUM_INFERENCE_RUNS = 5
RANDOM_SEED = 42
CLEANUP_SLEEP_SEC = 2
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def format_bytes(bytes_val: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} TB"


def get_timestamp() -> str:
    """Return formatted timestamp string."""
    return datetime.now().strftime(TIMESTAMP_FORMAT)


def get_memory_usage() -> int:
    """Get current GPU or CPU memory usage in bytes."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    else:
        import psutil

        return psutil.Process().memory_info().rss


def save_feature_map_png(tensor: torch.Tensor, save_path: Path):
    """Save first 16 channels of feature map as PNG grid."""
    if tensor.dim() != 4 or tensor.shape[1] == 0:
        return

    # Take first image in batch, first 16 channels
    num_channels = min(16, tensor.shape[1])
    features = tensor[0, :num_channels].cpu()

    # Create 4x4 grid
    grid_size = int(np.ceil(np.sqrt(num_channels)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten() if num_channels > 1 else [axes]

    for idx in range(num_channels):
        feat = features[idx].numpy()
        axes[idx].imshow(feat, cmap="viridis")
        axes[idx].axis("off")
        axes[idx].set_title(f"Ch {idx}", fontsize=8)

    # Hide unused subplots
    for idx in range(num_channels, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


def save_feature_map_npy(tensor: torch.Tensor, save_path: Path):
    """Save first 8 channels of feature map as NPY."""
    if tensor.dim() != 4 or tensor.shape[1] == 0:
        return

    # Take first image in batch, first 8 channels
    num_channels = min(8, tensor.shape[1])
    features = tensor[0, :num_channels].cpu().numpy()
    np.save(save_path, features)


class ModelBenchmark(ABC):
    """Abstract base class for model benchmarks."""

    MODEL_NAME: str
    INPUT_SIZE: int

    @abstractmethod
    def load(self):
        """Load model weights and prepare for inference."""
        pass

    @abstractmethod
    def infer(self, image_path: Path) -> float:
        """Run inference and return time in seconds."""
        pass

    @abstractmethod
    def dissect_layers(self, image_path: Path, viz_dir: Path) -> Dict:
        """Capture and visualize all model layers."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model and free memory."""
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        """Get model metadata."""
        pass


class DepthProBenchmark(ModelBenchmark):
    """DepthPro benchmark - metric depth estimation from HuggingFace."""

    # Configuration
    MODEL_NAME = "DepthPro"
    MODEL_ID = "apple/DepthPro-hf"
    INPUT_SIZE = 640

    # Uses HuggingFace transformers library
    # Auto-downloads model weights to cache (~1.2GB first time)
    # Provides metric depth estimation with field-of-view prediction

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

        print(f"[LOAD] Loading {self.MODEL_NAME} from {self.MODEL_ID}...")
        self.processor = DepthProImageProcessorFast.from_pretrained(self.MODEL_ID)
        self.model = DepthProForDepthEstimation.from_pretrained(
            self.MODEL_ID, use_fov_model=True
        ).to(self.device)
        self.model.eval()

    def infer(self, image_path: Path) -> float:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = self.model(**inputs)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return time.time() - start

    def dissect_layers(self, image_path: Path, viz_dir: Path) -> Dict:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        # Filter to only key feature-producing layers - very selective for large models
        all_layers = []
        for name, module in self.model.named_modules():
            # Skip container modules
            if len(list(module.children())) > 0:
                continue
            # Only capture Conv and key attention layers, skip norm/activation
            module_type = type(module).__name__
            if "conv" in module_type.lower():
                all_layers.append((name, module))
            elif (
                "attention" in module_type.lower()
                and "attention.attention" in name.lower()
            ):
                # Only main attention, not q/k/v projections
                all_layers.append((name, module))

        layers_metadata = []
        dissect_start = time.time()

        print(
            f"[DISSECT] Processing {len(all_layers)} key layers (Conv + Main Attention only)..."
        )

        for idx, (layer_name, layer_module) in enumerate(all_layers):
            activation = {}

            def hook(module, input, output):
                if torch.is_tensor(output):
                    activation["data"] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activation["data"] = output[0].detach().cpu()

            handle = layer_module.register_forward_hook(hook)

            with torch.no_grad():
                _ = self.model(**inputs)

            handle.remove()

            if "data" in activation and activation["data"] is not None:
                tensor = activation["data"]

                layer_info = {
                    "idx": idx,
                    "name": layer_name,
                    "type": type(layer_module).__name__,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "sparsity": float((tensor == 0).float().mean()),
                }
                layers_metadata.append(layer_info)

                if tensor.dim() == 4 and tensor.shape[1] > 0:
                    save_feature_map_png(tensor, viz_dir / f"layer_{idx:03d}.png")
                    save_feature_map_npy(tensor, viz_dir / f"layer_{idx:03d}.npy")

                if (idx + 1) % 20 == 0 or idx == len(all_layers) - 1:
                    elapsed = time.time() - dissect_start
                    print(
                        f"[DISSECT] {idx+1}/{len(all_layers)} layers | Elapsed: {elapsed:.1f}s | {datetime.now().strftime('%H:%M:%S')}"
                    )

                del tensor
                del activation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return {"layers": layers_metadata}

    def cleanup(self):
        del self.model
        del self.processor
        self.model = None
        self.processor = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.MODEL_NAME,
            "input_size": self.INPUT_SIZE,
            "total_params": total_params,
            "device": str(self.device),
        }


class DepthAnythingV2SmallBenchmark(ModelBenchmark):
    """Depth Anything V2 Small (ViT-S) - relative depth estimation."""

    # Configuration
    MODEL_NAME = "DepthAnythingV2-Small"
    VARIANT = "vits"
    INPUT_SIZE = 518
    FEATURES = 64
    OUT_CHANNELS = [48, 96, 192, 384]

    # Requires depth_anything_v2 module
    # Setup: git clone https://github.com/DepthAnything/Depth-Anything-V2
    # Then add to Python path or install as package

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    def load(self):
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(
                "depth_anything_v2 module not found. "
                "Please clone: git clone https://github.com/DepthAnything/Depth-Anything-V2"
            )

        print(f"[LOAD] Loading {self.MODEL_NAME} (variant: {self.VARIANT})...")
        self.model = DepthAnythingV2(
            encoder=self.VARIANT, features=self.FEATURES, out_channels=self.OUT_CHANNELS
        ).to(self.device)
        self.model.eval()

    def infer(self, image_path: Path) -> float:
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")

        # Resize and normalize
        transform = transforms.Compose(
            [
                transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        input_tensor = transform(image).unsqueeze(0).to(self.device)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            _ = self.model(input_tensor)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return time.time() - start

    def dissect_layers(self, image_path: Path, viz_dir: Path) -> Dict:
        from torchvision import transforms

        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        # Filter to only key feature-producing layers
        all_layers = []
        for name, module in self.model.named_modules():
            if len(list(module.children())) > 0:
                continue
            module_type = type(module).__name__
            # Only Conv and main Attention for depth models
            if "conv" in module_type.lower():
                all_layers.append((name, module))
            elif (
                "attention" in module_type.lower()
                and "attention.attention" in name.lower()
            ):
                all_layers.append((name, module))

        layers_metadata = []
        dissect_start = time.time()

        print(
            f"[DISSECT] Processing {len(all_layers)} key layers (Conv + Main Attention only)..."
        )

        for idx, (layer_name, layer_module) in enumerate(all_layers):
            activation = {}

            def hook(module, input, output):
                if torch.is_tensor(output):
                    activation["data"] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activation["data"] = output[0].detach().cpu()

            handle = layer_module.register_forward_hook(hook)

            with torch.no_grad():
                _ = self.model(input_tensor)

            handle.remove()

            if "data" in activation and activation["data"] is not None:
                tensor = activation["data"]

                layer_info = {
                    "idx": idx,
                    "name": layer_name,
                    "type": type(layer_module).__name__,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "sparsity": float((tensor == 0).float().mean()),
                }
                layers_metadata.append(layer_info)

                if tensor.dim() == 4 and tensor.shape[1] > 0:
                    save_feature_map_png(tensor, viz_dir / f"layer_{idx:03d}.png")
                    save_feature_map_npy(tensor, viz_dir / f"layer_{idx:03d}.npy")

                if (idx + 1) % 20 == 0 or idx == len(all_layers) - 1:
                    elapsed = time.time() - dissect_start
                    print(
                        f"[DISSECT] {idx+1}/{len(all_layers)} layers | Elapsed: {elapsed:.1f}s | {datetime.now().strftime('%H:%M:%S')}"
                    )

                del tensor
                del activation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return {"layers": layers_metadata}

    def cleanup(self):
        del self.model
        self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.MODEL_NAME,
            "input_size": self.INPUT_SIZE,
            "total_params": total_params,
            "device": str(self.device),
        }


class DepthAnythingV2BaseBenchmark(DepthAnythingV2SmallBenchmark):
    """Depth Anything V2 Base (ViT-B) - relative depth estimation."""

    MODEL_NAME = "DepthAnythingV2-Base"
    VARIANT = "vitb"
    FEATURES = 128
    OUT_CHANNELS = [96, 192, 384, 768]


class DepthAnythingV2LargeBenchmark(DepthAnythingV2SmallBenchmark):
    """Depth Anything V2 Large (ViT-L) - relative depth estimation."""

    MODEL_NAME = "DepthAnythingV2-Large"
    VARIANT = "vitl"
    FEATURES = 256
    OUT_CHANNELS = [256, 512, 1024, 1024]


class YOLO11nDetectBenchmark(ModelBenchmark):
    """YOLO11n Detection - object detection."""

    # Configuration
    MODEL_NAME = "YOLO11n-Detect"
    MODEL_PATH = "models/yolo11n.pt"
    INPUT_SIZE = 640

    # Uses Ultralytics library
    # Auto-downloads if models/yolo11n.pt missing

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    def load(self):
        from ultralytics import YOLO

        print(f"[LOAD] Loading {self.MODEL_NAME} from {self.MODEL_PATH}...")
        self.model = YOLO(self.MODEL_PATH)
        self.model.to(self.device)

    def infer(self, image_path: Path) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        _ = self.model(str(image_path), verbose=False)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return time.time() - start

    def dissect_layers(self, image_path: Path, viz_dir: Path) -> Dict:
        # Get model's underlying PyTorch model
        pt_model = self.model.model

        # Filter to only meaningful layers
        all_layers = []
        for name, module in pt_model.named_modules():
            if len(list(module.children())) > 0:
                continue
            module_type = type(module).__name__
            if any(
                x in module_type.lower()
                for x in ["conv", "linear", "norm", "relu", "silu", "bottleneck"]
            ):
                all_layers.append((name, module))

        layers_metadata = []
        dissect_start = time.time()

        print(
            f"[DISSECT] Processing {len(all_layers)} meaningful layers (filtered from containers)..."
        )

        # Prepare input
        from PIL import Image as PILImage
        import torchvision.transforms as transforms

        image = PILImage.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
                transforms.ToTensor(),
            ]
        )
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        for idx, (layer_name, layer_module) in enumerate(all_layers):
            activation = {}

            def hook(module, input, output):
                if torch.is_tensor(output):
                    activation["data"] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activation["data"] = output[0].detach().cpu()

            handle = layer_module.register_forward_hook(hook)

            with torch.no_grad():
                try:
                    _ = pt_model(input_tensor)
                except:
                    pass

            handle.remove()

            if "data" in activation and activation["data"] is not None:
                tensor = activation["data"]

                layer_info = {
                    "idx": idx,
                    "name": layer_name,
                    "type": type(layer_module).__name__,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "sparsity": float((tensor == 0).float().mean()),
                }
                layers_metadata.append(layer_info)

                if tensor.dim() == 4 and tensor.shape[1] > 0:
                    save_feature_map_png(tensor, viz_dir / f"layer_{idx:03d}.png")
                    save_feature_map_npy(tensor, viz_dir / f"layer_{idx:03d}.npy")

                if (idx + 1) % 20 == 0 or idx == len(all_layers) - 1:
                    elapsed = time.time() - dissect_start
                    print(
                        f"[DISSECT] {idx+1}/{len(all_layers)} layers | Elapsed: {elapsed:.1f}s | {datetime.now().strftime('%H:%M:%S')}"
                    )

                del tensor
                del activation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return {"layers": layers_metadata}

    def cleanup(self):
        del self.model
        self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_info(self) -> Dict:
        pt_model = self.model.model
        total_params = sum(p.numel() for p in pt_model.parameters())
        return {
            "model_name": self.MODEL_NAME,
            "input_size": self.INPUT_SIZE,
            "total_params": total_params,
            "device": str(self.device),
        }


class YOLO11nSegmentBenchmark(YOLO11nDetectBenchmark):
    """YOLO11n Segmentation - instance segmentation."""

    MODEL_NAME = "YOLO11n-Segment"
    MODEL_PATH = "models/yolo11n-seg.pt"


class YOLO11nPoseBenchmark(YOLO11nDetectBenchmark):
    """YOLO11n Pose - keypoint detection."""

    MODEL_NAME = "YOLO11n-Pose"
    MODEL_PATH = "models/yolo11n-pose.pt"


class MobileSAMBenchmark(ModelBenchmark):
    """Mobile SAM - segment anything model."""

    # Configuration
    MODEL_NAME = "MobileSAM"
    MODEL_PATH = "models/mobile_sam.pt"
    INPUT_SIZE = 1024

    # Uses Ultralytics library

    def __init__(self, device: torch.device):
        self.device = device
        self.model = None

    def load(self):
        from ultralytics import SAM

        print(f"[LOAD] Loading {self.MODEL_NAME} from {self.MODEL_PATH}...")
        self.model = SAM(self.MODEL_PATH)
        self.model.to(self.device)

    def infer(self, image_path: Path) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        _ = self.model(str(image_path), verbose=False)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        return time.time() - start

    def dissect_layers(self, image_path: Path, viz_dir: Path) -> Dict:
        pt_model = self.model.model

        # Filter to only meaningful layers
        all_layers = []
        for name, module in pt_model.named_modules():
            if len(list(module.children())) > 0:
                continue
            module_type = type(module).__name__
            if any(
                x in module_type.lower()
                for x in ["conv", "linear", "attention", "norm", "relu", "gelu"]
            ):
                all_layers.append((name, module))

        layers_metadata = []
        dissect_start = time.time()

        print(
            f"[DISSECT] Processing {len(all_layers)} meaningful layers (filtered from containers)..."
        )

        from PIL import Image as PILImage
        import torchvision.transforms as transforms

        image = PILImage.open(image_path).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
                transforms.ToTensor(),
            ]
        )
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        for idx, (layer_name, layer_module) in enumerate(all_layers):
            activation = {}

            def hook(module, input, output):
                if torch.is_tensor(output):
                    activation["data"] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    if torch.is_tensor(output[0]):
                        activation["data"] = output[0].detach().cpu()

            handle = layer_module.register_forward_hook(hook)

            with torch.no_grad():
                try:
                    _ = pt_model.image_encoder(input_tensor)
                except:
                    pass

            handle.remove()

            if "data" in activation and activation["data"] is not None:
                tensor = activation["data"]

                layer_info = {
                    "idx": idx,
                    "name": layer_name,
                    "type": type(layer_module).__name__,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "min": float(tensor.min()),
                    "max": float(tensor.max()),
                    "mean": float(tensor.mean()),
                    "std": float(tensor.std()),
                    "sparsity": float((tensor == 0).float().mean()),
                }
                layers_metadata.append(layer_info)

                if tensor.dim() == 4 and tensor.shape[1] > 0:
                    save_feature_map_png(tensor, viz_dir / f"layer_{idx:03d}.png")
                    save_feature_map_npy(tensor, viz_dir / f"layer_{idx:03d}.npy")

                if (idx + 1) % 20 == 0 or idx == len(all_layers) - 1:
                    elapsed = time.time() - dissect_start
                    print(
                        f"[DISSECT] {idx+1}/{len(all_layers)} layers | Elapsed: {elapsed:.1f}s | {datetime.now().strftime('%H:%M:%S')}"
                    )

                del tensor
                del activation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return {"layers": layers_metadata}

    def cleanup(self):
        del self.model
        self.model = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    def get_info(self) -> Dict:
        pt_model = self.model.model
        total_params = sum(p.numel() for p in pt_model.parameters())
        return {
            "model_name": self.MODEL_NAME,
            "input_size": self.INPUT_SIZE,
            "total_params": total_params,
            "device": str(self.device),
        }


def generate_reports(
    results: List[Dict],
    timestamp: str,
    total_time: float,
    device: torch.device,
    test_images: List[Path],
):
    """Generate JSON, CSV, and Markdown reports."""
    results_dir = Path("vision-bench/results")

    # JSON - Complete raw data
    json_path = results_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "total_execution_time_sec": total_time,
                "device": str(device),
                "random_seed": RANDOM_SEED,
                "num_inference_runs": NUM_INFERENCE_RUNS,
                "results": results,
            },
            f,
            indent=2,
        )

    # CSV - Summary table
    csv_path = results_dir / f"benchmark_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_name",
                "load_time_sec",
                "avg_inference_sec",
                "std_inference_sec",
                "min_inference_sec",
                "max_inference_sec",
                "fps_avg",
                "fps_std",
                "mem_peak_mb",
                "total_params",
                "total_layers_dissected",
                "num_images",
                "num_runs_per_image",
                "device",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in writer.fieldnames})

    # Markdown - Readable report
    md_path = results_dir / f"benchmark_report_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(f"# Vision Model Benchmark Report\n\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")

        # System info
        f.write(f"## System Information\n\n")
        f.write(f"| Property | Value |\n")
        f.write(f"|----------|-------|\n")
        f.write(f"| OS | {platform.system()} {platform.release()} |\n")
        f.write(f"| Python | {platform.python_version()} |\n")
        f.write(f"| PyTorch | {torch.__version__} |\n")
        f.write(f"| CUDA Available | {torch.cuda.is_available()} |\n")
        if torch.cuda.is_available():
            f.write(f"| CUDA Version | {torch.version.cuda} |\n")
            f.write(f"| GPU | {torch.cuda.get_device_name(0)} |\n")
        f.write(f"| Device | {device} |\n")
        f.write(f"| Random Seed | {RANDOM_SEED} |\n\n")

        # Configuration
        f.write(f"## Benchmark Configuration\n\n")
        f.write(f"- Test Images: {len(test_images)}\n")
        f.write(f"- Inference Runs per Image: {NUM_INFERENCE_RUNS}\n")
        f.write(
            f"- Total Inference Runs per Model: {len(test_images) * NUM_INFERENCE_RUNS}\n"
        )
        f.write(f"- Total Execution Time: {total_time:.2f}s\n\n")

        # Results table
        f.write(f"## Results\n\n")
        f.write(
            f"| Model | Load Time (s) | Avg Inference (s) | Std (s) | FPS | Peak Memory (MB) | Parameters | Layers |\n"
        )
        f.write(
            f"|-------|---------------|-------------------|---------|-----|------------------|------------|--------|\n"
        )
        for r in results:
            f.write(
                f"| {r['model_name']} | {r['load_time_sec']:.2f} | "
                f"{r['avg_inference_sec']:.3f} | {r['std_inference_sec']:.3f} | "
                f"{r['fps_avg']:.2f} | {r['mem_peak_mb']:.0f} | "
                f"{r['total_params']:,} | {r['total_layers_dissected']} |\n"
            )

        # Analysis
        f.write(f"\n## Analysis\n\n")

        fastest = min(results, key=lambda x: x["avg_inference_sec"])
        slowest = max(results, key=lambda x: x["avg_inference_sec"])
        most_memory = max(results, key=lambda x: x["mem_peak_mb"])
        most_params = max(results, key=lambda x: x["total_params"])

        f.write(
            f"- **Fastest Model:** {fastest['model_name']} ({fastest['avg_inference_sec']:.3f}s, {fastest['fps_avg']:.2f} FPS)\n"
        )
        f.write(
            f"- **Slowest Model:** {slowest['model_name']} ({slowest['avg_inference_sec']:.3f}s, {slowest['fps_avg']:.2f} FPS)\n"
        )
        f.write(
            f"- **Most Memory Intensive:** {most_memory['model_name']} ({most_memory['mem_peak_mb']:.0f} MB)\n"
        )
        f.write(
            f"- **Most Parameters:** {most_params['model_name']} ({most_params['total_params']:,})\n\n"
        )

        # Visualization links
        f.write(f"## Visualizations\n\n")
        f.write(f"Layer dissection visualizations are saved in:\n\n")
        for r in results:
            f.write(f"- `vision-bench/viz/{timestamp}/{r['model_name']}/`\n")

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"Models Benchmarked: {len(results)}")
    print(f"Test Images: {len(test_images)}")
    print(f"Inference Runs per Image: {NUM_INFERENCE_RUNS}")
    print(
        f"Total Inference Runs: {len(test_images) * NUM_INFERENCE_RUNS * len(results)}"
    )
    print(f"Total Execution Time: {total_time:.2f}s")
    print()
    print("Output Files:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {md_path}")
    print()
    print("Visualizations:")
    print(f"  - vision-bench/viz/{timestamp}/")
    print("=" * 80)


def run_benchmark():
    """Main benchmark execution."""
    # Set random seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] Using CUDA device")
    else:
        device = torch.device("cpu")
        print("[WARNING] CUDA not available, falling back to CPU")

    # Find test images
    test_images = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        test_images.extend(Path("test_data").glob(ext))

    if not test_images:
        print("[ERROR] No test images found in test_data/")
        sys.exit(1)

    # Validate images
    validated_images = []
    for img_path in test_images:
        try:
            img = Image.open(img_path)
            img.verify()
            validated_images.append(img_path)
        except Exception as e:
            print(f"[WARNING] Skipping invalid image {img_path}: {e}")

    test_images = validated_images
    print(f"[INFO] Found {len(test_images)} valid test images")

    # Create timestamped output directories
    timestamp = get_timestamp()
    viz_dir = Path(f"vision-bench/viz/{timestamp}")
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Benchmark classes ordered from smallest to largest
    benchmark_classes = [
        DepthProBenchmark,  # ~1.3B params
        YOLO11nDetectBenchmark,  # ~3M params
        YOLO11nSegmentBenchmark,  # ~3M params
        YOLO11nPoseBenchmark,  # ~3M params
        MobileSAMBenchmark,  # ~9M params
        DepthAnythingV2SmallBenchmark,  # ~24M params
        DepthAnythingV2BaseBenchmark,  # ~97M params
        DepthAnythingV2LargeBenchmark,  # ~335M params
    ]

    all_results = []
    total_start = time.time()

    for model_idx, BenchmarkClass in enumerate(benchmark_classes, 1):
        print("\n" + "=" * 80)
        print(f"[{model_idx}/{len(benchmark_classes)}] {BenchmarkClass.MODEL_NAME}")
        print("=" * 80)

        try:
            # Initialize benchmark
            benchmark = BenchmarkClass(device)

            # Memory before load
            mem_before = get_memory_usage()
            print(f"[MEMORY] Before load: {format_bytes(mem_before)}")

            # Load model
            load_start = time.time()
            benchmark.load()
            load_time = time.time() - load_start

            mem_after_load = get_memory_usage()
            print(f"[LOAD] Model loaded in {load_time:.2f}s")
            print(
                f"[MEMORY] After load: {format_bytes(mem_after_load)} "
                f"(delta: +{format_bytes(mem_after_load - mem_before)})"
            )

            # Run inference on all images
            all_inference_times = []
            mem_peak = mem_after_load

            for img_idx, image_path in enumerate(test_images, 1):
                for run_idx in range(NUM_INFERENCE_RUNS):
                    inference_time = benchmark.infer(image_path)
                    all_inference_times.append(inference_time)

                    current_mem = get_memory_usage()
                    mem_peak = max(mem_peak, current_mem)

                    total_runs = len(test_images) * NUM_INFERENCE_RUNS
                    current_run = (img_idx - 1) * NUM_INFERENCE_RUNS + run_idx + 1

                    print(
                        f"[INFER {current_run}/{total_runs}] {image_path.name} | "
                        f"Time: {inference_time:.3f}s | Memory: {format_bytes(current_mem)}"
                    )

            # Calculate statistics
            avg_time = np.mean(all_inference_times)
            std_time = np.std(all_inference_times)
            min_time = np.min(all_inference_times)
            max_time = np.max(all_inference_times)
            fps_avg = 1.0 / avg_time if avg_time > 0 else 0
            fps_std = fps_avg * (std_time / avg_time) if avg_time > 0 else 0

            print(
                f"[STATS] Inference: avg={avg_time:.3f}s std={std_time:.3f}s "
                f"min={min_time:.3f}s max={max_time:.3f}s"
            )
            print(f"[STATS] FPS: avg={fps_avg:.2f} std={fps_std:.2f}")

            # Get model info before dissection
            model_info = benchmark.get_info()

            # Layer dissection on first image
            print(f"[DISSECT] Starting layer dissection on {test_images[0].name}...")
            print(f"[DISSECT] Model: {BenchmarkClass.MODEL_NAME}")
            print(f"[DISSECT] Parameters: {model_info['total_params']:,}")
            print(f"[DISSECT] Start time: {datetime.now().strftime('%H:%M:%S')}")

            model_viz_dir = viz_dir / BenchmarkClass.MODEL_NAME
            model_viz_dir.mkdir(exist_ok=True)

            dissect_start = time.time()
            layer_info = benchmark.dissect_layers(test_images[0], model_viz_dir)
            dissect_time = time.time() - dissect_start

            print(
                f"[DISSECT] Completed in {dissect_time:.2f}s ({dissect_time/60:.1f}m)"
            )

            # Save layer metadata
            with open(model_viz_dir / "layers_metadata.json", "w") as f:
                json.dump(layer_info, f, indent=2)

            print(f"[DISSECT] Saved {len(layer_info['layers'])} layer visualizations")

            # Cleanup
            print(f"[CLEANUP] Clearing model and cache...")
            benchmark.cleanup()
            time.sleep(CLEANUP_SLEEP_SEC)

            mem_after_cleanup = get_memory_usage()
            print(f"[MEMORY] After cleanup: {format_bytes(mem_after_cleanup)}")

            # Store results
            result = {
                "model_name": BenchmarkClass.MODEL_NAME,
                "load_time_sec": load_time,
                "inference_times_sec": all_inference_times,
                "avg_inference_sec": avg_time,
                "std_inference_sec": std_time,
                "min_inference_sec": min_time,
                "max_inference_sec": max_time,
                "fps_avg": fps_avg,
                "fps_std": fps_std,
                "mem_before_mb": mem_before / (1024**2),
                "mem_after_load_mb": mem_after_load / (1024**2),
                "mem_peak_mb": mem_peak / (1024**2),
                "mem_after_cleanup_mb": mem_after_cleanup / (1024**2),
                "total_params": model_info["total_params"],
                "total_layers_dissected": len(layer_info["layers"]),
                "num_images": len(test_images),
                "num_runs_per_image": NUM_INFERENCE_RUNS,
                "input_size": BenchmarkClass.INPUT_SIZE,
                "device": str(device),
            }
            all_results.append(result)

            print(f"[COMPLETE] {BenchmarkClass.MODEL_NAME} finished")

        except Exception as e:
            print(f"[ERROR] Failed to benchmark {BenchmarkClass.MODEL_NAME}")
            print(f"[ERROR] {str(e)}")
            traceback.print_exc()
            sys.exit(1)

    total_time = time.time() - total_start

    # Generate reports
    generate_reports(all_results, timestamp, total_time, device, test_images)


if __name__ == "__main__":
    run_benchmark()
