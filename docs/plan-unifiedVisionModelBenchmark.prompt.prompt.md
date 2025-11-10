# Unified Vision Model Benchmark System - Complete Specification

## Overview
A single-file benchmarking script `vision-bench/unified_benchmark.py` that sequentially evaluates all vision models on all test images with comprehensive layer dissection, memory-efficient processing, and automatic CPU fallback.

## Architecture Decisions

### Configuration
- **Hardcoded configs** in each model class for easy developer modification
- **Sequential execution** only (no parallel mode)
- **Batch size = 1** for pure statistical measurement
- **Native input sizes** per model (no multi-resolution testing)
- **Reproducibility** with fixed random seed (42)
- **Auto-download** for model weights with progress tracking
- **No warmup runs** - pure cold-start metrics including CUDA initialization

### Execution Flow
- **Sequential processing** of all models to prevent GPU exhaustion
- **Fail-fast** error handling - exit on first failure with full traceback
- **All test images** processed per model with aggregated statistics
- **Device fallback** to CPU if CUDA unavailable with warning message
- **Detailed progress logging** with real-time status updates

### Layer Dissection
- **All layers captured** (no filtering or limits)
- **Memory-efficient** approach: process one layer at a time (hook→capture→visualize→save→delete→next)
- **Dual output formats**: PNG visualizations + NPY arrays for later analysis
- **Layer naming**: Use layer index only (layer_001.png) for consistency
- **Single folder** per model (no pagination regardless of layer count)

### Output & Reporting
- **Three output formats**: JSON (raw data), CSV (summary), Markdown (report)
- **Simple black/white formatting** - no colors, no emojis
- **Terminal logging** with detailed progress indicators
- **Timestamp format**: YYYYMMDD_HHMMSS for all file/folder naming

### Dependencies
- **Depth Anything V2**: Include inline setup comments, skip if unavailable with clear error message
- **Existing models directory**: Use `../models/` for YOLO and SAM weights
- **Auto-directory creation**: Create all necessary output folders automatically

## Constants

```python
NUM_INFERENCE_RUNS = 5           # Number of inference runs per image
RANDOM_SEED = 42                 # For reproducibility
CLEANUP_SLEEP_SEC = 2            # Sleep duration between models for GPU cleanup
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"  # Folder/file naming format
```

## Model Registry (8 Models)

### 1. DepthProBenchmark
- **Model ID**: `"apple/DepthPro-hf"`
- **Input Size**: 640x640
- **Library**: HuggingFace Transformers
- **Auto-download**: Yes (to cache, ~1.2GB)
- **Notes**: Metric depth estimation with FOV prediction

### 2. DepthAnythingV2SmallBenchmark
- **Variant**: vits
- **Input Size**: 518
- **Features**: 64
- **Out Channels**: [48, 96, 192, 384]
- **Dependency**: `depth_anything_v2` module
- **Setup Comment**: `# git clone https://github.com/DepthAnything/Depth-Anything-V2`

### 3. DepthAnythingV2BaseBenchmark
- **Variant**: vitb
- **Input Size**: 518
- **Features**: 128
- **Out Channels**: [96, 192, 384, 768]

### 4. DepthAnythingV2LargeBenchmark
- **Variant**: vitl
- **Input Size**: 518
- **Features**: 256
- **Out Channels**: [256, 512, 1024, 1024]

### 5. YOLO11nDetectBenchmark
- **Model Path**: `"../models/yolo11n.pt"`
- **Input Size**: 640x640
- **Library**: Ultralytics
- **Auto-download**: Yes if missing from models/

### 6. YOLO11nSegmentBenchmark
- **Model Path**: `"../models/yolo11n-seg.pt"`
- **Input Size**: 640x640

### 7. YOLO11nPoseBenchmark
- **Model Path**: `"../models/yolo11n-pose.pt"`
- **Input Size**: 640x640

### 8. MobileSAMBenchmark
- **Model Path**: `"../models/mobile_sam.pt"`
- **Input Size**: 1024x1024
- **Library**: Ultralytics

## Implementation Steps

### Step 1: Script Foundation & Utilities

**Initialize Environment:**
```python
# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Detect device with fallback
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[INFO] Using CUDA device")
else:
    device = torch.device("cpu")
    print("[WARNING] CUDA not available, falling back to CPU. Performance will be significantly slower.")
```

**Create Output Directories:**
```python
Path("vision-bench").mkdir(exist_ok=True)
Path("vision-bench/results").mkdir(exist_ok=True)
Path("vision-bench/viz").mkdir(exist_ok=True)
```

**Image Discovery:**
```python
test_images = []
for ext in ["*.jpg", "*.png"]:
    test_images.extend(Path("test_data").glob(ext))

# Validate images are readable
validated_images = []
for img_path in test_images:
    try:
        Image.open(img_path).verify()
        validated_images.append(img_path)
    except Exception as e:
        print(f"[WARNING] Skipping invalid image {img_path}: {e}")

print(f"[INFO] Found {len(validated_images)} valid test images")
```

**Utility Functions:**
- `format_bytes(bytes)` - Convert bytes to human-readable (MB/GB)
- `get_timestamp()` - Return formatted timestamp string
- `log(message, level)` - Standardized logging with timestamp
- `save_feature_map_png(tensor, path)` - Save first 16 channels as PNG grid
- `save_feature_map_npy(tensor, path)` - Save first 8 channels as NPY
- `sanitize_layer_name(name)` - Not used, but available if needed

**Abstract Base Class:**
```python
class ModelBenchmark(ABC):
    """
    Abstract base class for model benchmarks.
    
    Each model implementation must:
    1. Define MODEL_NAME, INPUT_SIZE, and other config as class attributes
    2. Implement load() to initialize the model
    3. Implement infer(image_path) to run inference and return time
    4. Implement dissect_layers(image_path) to capture and visualize all layers
    5. Implement cleanup() to free GPU/CPU memory
    6. Implement get_info() to return model metadata
    """
    
    @abstractmethod
    def load(self):
        """Load model weights and prepare for inference."""
        pass
    
    @abstractmethod
    def infer(self, image_path: Path) -> float:
        """
        Run inference on image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Inference time in seconds
        """
        pass
    
    @abstractmethod
    def dissect_layers(self, image_path: Path) -> dict:
        """
        Capture and visualize all model layers.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Dictionary with layer metadata
        """
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up model and free memory."""
        pass
    
    @abstractmethod
    def get_info(self) -> dict:
        """
        Get model information.
        
        Returns:
            Dictionary with model metadata (params, layers, etc.)
        """
        pass
```

### Step 2: Model Benchmark Implementations

**Each model class structure:**
```python
class DepthProBenchmark(ModelBenchmark):
    # Hardcoded configuration - modify these values to change model behavior
    MODEL_NAME = "DepthPro"
    MODEL_ID = "apple/DepthPro-hf"  # HuggingFace model identifier
    INPUT_SIZE = 640  # Native input size for optimal performance
    
    # Uses HuggingFace transformers library
    # Auto-downloads model weights to cache (~1.2GB first time)
    # Provides metric depth estimation with field-of-view prediction
    
    def __init__(self, device):
        self.device = device
        self.model = None
        self.processor = None
    
    def load(self):
        """Load DepthPro model from HuggingFace."""
        from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation
        
        print(f"[LOAD] Loading {self.MODEL_NAME} from {self.MODEL_ID}...")
        self.processor = DepthProImageProcessorFast.from_pretrained(self.MODEL_ID)
        self.model = DepthProForDepthEstimation.from_pretrained(
            self.MODEL_ID, 
            use_fov_model=True
        ).to(self.device)
        self.model.eval()
    
    def infer(self, image_path: Path) -> float:
        """Run inference and return timing."""
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Time inference with CUDA synchronization for accuracy
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        return elapsed
    
    def dissect_layers(self, image_path: Path) -> dict:
        """Capture all layers one-by-one for memory efficiency."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Get all layers
        all_layers = list(self.model.named_modules())
        layers_metadata = []
        
        print(f"[DISSECT] Processing {len(all_layers)} layers...")
        
        for idx, (layer_name, layer_module) in enumerate(all_layers):
            # Skip container modules (Sequential, ModuleList, etc.)
            if len(list(layer_module.children())) > 0:
                continue
            
            activation = {}
            
            def hook(module, input, output):
                if torch.is_tensor(output):
                    activation['data'] = output.detach().cpu()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    activation['data'] = output[0].detach().cpu() if torch.is_tensor(output[0]) else None
            
            # Register hook, run inference, capture
            handle = layer_module.register_forward_hook(hook)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            handle.remove()
            
            # Process activation if captured
            if 'data' in activation and activation['data'] is not None:
                tensor = activation['data']
                
                # Compute statistics
                layer_info = {
                    'idx': idx,
                    'name': layer_name,
                    'type': type(layer_module).__name__,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'min': float(tensor.min()),
                    'max': float(tensor.max()),
                    'mean': float(tensor.mean()),
                    'std': float(tensor.std()),
                    'sparsity': float((tensor == 0).float().mean())
                }
                layers_metadata.append(layer_info)
                
                # Save visualizations if 4D tensor
                if tensor.dim() == 4 and tensor.shape[1] > 0:
                    save_feature_map_png(tensor, f"layer_{idx:03d}.png")
                    save_feature_map_npy(tensor, f"layer_{idx:03d}.npy")
                
                # Print progress
                print(f"[DISSECT {idx+1}/{len(all_layers)}] Layer {idx:03d}: {tensor.shape}")
                
                # Free memory immediately
                del tensor
                del activation
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
        
        return {'layers': layers_metadata}
    
    def cleanup(self):
        """Free model memory."""
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def get_info(self) -> dict:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_name': self.MODEL_NAME,
            'input_size': self.INPUT_SIZE,
            'total_params': total_params,
            'device': str(self.device)
        }
```

**Similar implementations for:**
- `DepthAnythingV2SmallBenchmark` with inline comment: `# Requires depth_anything_v2 module: git clone https://github.com/DepthAnything/Depth-Anything-V2`
- `DepthAnythingV2BaseBenchmark`
- `DepthAnythingV2LargeBenchmark`
- `YOLO11nDetectBenchmark` with inline comment: `# Uses Ultralytics library, auto-downloads if models/yolo11n.pt missing`
- `YOLO11nSegmentBenchmark`
- `YOLO11nPoseBenchmark`
- `MobileSAMBenchmark`

### Step 3: Memory-Efficient Layer Dissection

**Key Design Principles:**
1. Process one layer at a time to minimize memory usage
2. Register hook → run inference → capture → save → delete → next
3. Save both PNG (visualization) and NPY (raw data) formats
4. Use layer index for consistent naming: `layer_001.png`, `layer_001.npy`
5. Store layer mapping in `layers_metadata.json`

**Dissection Algorithm:**
```python
def dissect_layers(self, image_path: Path) -> dict:
    # Get all module layers
    all_layers = list(self.model.named_modules())
    
    # Filter out container layers (only leaf modules)
    leaf_layers = [(name, mod) for name, mod in all_layers 
                   if len(list(mod.children())) == 0]
    
    layers_metadata = []
    
    for idx, (layer_name, layer_module) in enumerate(leaf_layers):
        # Single hook for this layer only
        activation = {}
        
        def hook(module, input, output):
            # Capture output, handle tuples
            if torch.is_tensor(output):
                activation['data'] = output.detach().cpu()
            elif isinstance(output, (tuple, list)):
                activation['data'] = output[0].detach().cpu()
        
        handle = layer_module.register_forward_hook(hook)
        
        # Run inference
        with torch.no_grad():
            _ = self.model(preprocessed_input)
        
        # Remove hook immediately
        handle.remove()
        
        # Process captured activation
        if 'data' in activation:
            tensor = activation['data']
            
            # Compute statistics
            stats = {
                'idx': idx,
                'name': layer_name,
                'type': type(layer_module).__name__,
                'shape': list(tensor.shape),
                'min': float(tensor.min()),
                'max': float(tensor.max()),
                'mean': float(tensor.mean()),
                'std': float(tensor.std())
            }
            layers_metadata.append(stats)
            
            # Save if 4D tensor (B, C, H, W)
            if tensor.dim() == 4:
                # PNG: first 16 channels as grid
                save_feature_map_png(tensor, f"layer_{idx:03d}.png")
                # NPY: first 8 channels for analysis
                save_feature_map_npy(tensor[:, :8], f"layer_{idx:03d}.npy")
            
            # Clean up immediately
            del tensor
            del activation
            torch.cuda.empty_cache()
        
        print(f"[DISSECT {idx+1}/{len(leaf_layers)}] Layer {idx:03d}")
    
    return {'layers': layers_metadata}
```

### Step 4: Sequential Execution Engine

**Main Benchmark Loop:**
```python
def run_benchmark():
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    
    # Create timestamped output directories
    results_dir = Path("vision-bench/results")
    viz_dir = Path(f"vision-bench/viz/{timestamp}")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all benchmark classes
    benchmark_classes = [
        DepthProBenchmark,
        DepthAnythingV2SmallBenchmark,
        DepthAnythingV2BaseBenchmark,
        DepthAnythingV2LargeBenchmark,
        YOLO11nDetectBenchmark,
        YOLO11nSegmentBenchmark,
        YOLO11nPoseBenchmark,
        MobileSAMBenchmark
    ]
    
    all_results = []
    total_start = time.time()
    
    for model_idx, BenchmarkClass in enumerate(benchmark_classes, 1):
        print("=" * 80)
        print(f"[{model_idx}/{len(benchmark_classes)}] {BenchmarkClass.MODEL_NAME}")
        print("=" * 80)
        
        try:
            # Initialize benchmark
            benchmark = BenchmarkClass(device)
            
            # Memory snapshot before load
            mem_before = get_memory_usage()
            print(f"[MEMORY] Before load: {format_bytes(mem_before)}")
            
            # Load model
            load_start = time.time()
            benchmark.load()
            load_time = time.time() - load_start
            
            mem_after_load = get_memory_usage()
            print(f"[LOAD] Model loaded in {load_time:.2f}s")
            print(f"[MEMORY] After load: {format_bytes(mem_after_load)} (delta: +{format_bytes(mem_after_load - mem_before)})")
            
            # Run inference on all images
            all_inference_times = []
            
            for img_idx, image_path in enumerate(test_images, 1):
                for run_idx in range(NUM_INFERENCE_RUNS):
                    mem_before_infer = get_memory_usage()
                    
                    inference_time = benchmark.infer(image_path)
                    all_inference_times.append(inference_time)
                    
                    mem_peak = get_memory_usage()
                    
                    total_runs = len(test_images) * NUM_INFERENCE_RUNS
                    current_run = (img_idx - 1) * NUM_INFERENCE_RUNS + run_idx + 1
                    
                    print(f"[INFER {current_run}/{total_runs}] {image_path.name} | "
                          f"Time: {inference_time:.3f}s | Peak Memory: {format_bytes(mem_peak)}")
            
            # Calculate statistics
            avg_time = np.mean(all_inference_times)
            std_time = np.std(all_inference_times)
            min_time = np.min(all_inference_times)
            max_time = np.max(all_inference_times)
            fps_avg = 1.0 / avg_time if avg_time > 0 else 0
            fps_std = fps_avg * (std_time / avg_time) if avg_time > 0 else 0
            
            print(f"[STATS] Inference: avg={avg_time:.3f}s std={std_time:.3f}s min={min_time:.3f}s max={max_time:.3f}s")
            print(f"[STATS] FPS: avg={fps_avg:.2f} std={fps_std:.2f}")
            
            # Layer dissection on first image
            print(f"[DISSECT] Starting layer dissection...")
            model_viz_dir = viz_dir / BenchmarkClass.MODEL_NAME
            model_viz_dir.mkdir(exist_ok=True)
            
            layer_info = benchmark.dissect_layers(test_images[0])
            
            # Save layer metadata
            with open(model_viz_dir / "layers_metadata.json", "w") as f:
                json.dump(layer_info, f, indent=2)
            
            print(f"[DISSECT] Saved {len(layer_info['layers'])} layer visualizations")
            
            # Cleanup
            print(f"[CLEANUP] Clearing model and GPU cache...")
            benchmark.cleanup()
            time.sleep(CLEANUP_SLEEP_SEC)
            
            mem_after_cleanup = get_memory_usage()
            print(f"[MEMORY] After cleanup: {format_bytes(mem_after_cleanup)}")
            
            # Store results
            model_info = benchmark.get_info()
            result = {
                'model_name': BenchmarkClass.MODEL_NAME,
                'load_time_sec': load_time,
                'inference_times_sec': all_inference_times,
                'avg_inference_sec': avg_time,
                'std_inference_sec': std_time,
                'min_inference_sec': min_time,
                'max_inference_sec': max_time,
                'fps_avg': fps_avg,
                'fps_std': fps_std,
                'mem_before_mb': mem_before / (1024**2),
                'mem_after_load_mb': mem_after_load / (1024**2),
                'mem_peak_mb': mem_peak / (1024**2),
                'mem_after_cleanup_mb': mem_after_cleanup / (1024**2),
                'total_params': model_info['total_params'],
                'total_layers_dissected': len(layer_info['layers']),
                'num_images': len(test_images),
                'num_runs_per_image': NUM_INFERENCE_RUNS,
                'input_size': BenchmarkClass.INPUT_SIZE,
                'device': str(device)
            }
            all_results.append(result)
            
            print(f"[COMPLETE] {BenchmarkClass.MODEL_NAME} finished")
            
        except Exception as e:
            print(f"[ERROR] Failed to benchmark {BenchmarkClass.MODEL_NAME}")
            print(f"[ERROR] Exception: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    total_time = time.time() - total_start
    
    # Generate reports
    generate_reports(all_results, timestamp, total_time)
```

### Step 5: Report Generation

**Generate Three Output Formats:**

```python
def generate_reports(results, timestamp, total_time):
    results_dir = Path("vision-bench/results")
    
    # 1. JSON - Complete raw data
    json_path = results_dir / f"benchmark_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            'timestamp': timestamp,
            'total_execution_time_sec': total_time,
            'device': str(device),
            'random_seed': RANDOM_SEED,
            'num_inference_runs': NUM_INFERENCE_RUNS,
            'results': results
        }, f, indent=2)
    
    # 2. CSV - Summary table
    csv_path = results_dir / f"benchmark_summary_{timestamp}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            'model_name', 'load_time_sec', 'avg_inference_sec', 'std_inference_sec',
            'min_inference_sec', 'max_inference_sec', 'fps_avg', 'fps_std',
            'mem_peak_mb', 'total_params', 'total_layers_dissected',
            'num_images', 'num_runs_per_image', 'device'
        ])
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in writer.fieldnames})
    
    # 3. Markdown - Readable report
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
        f.write(f"- Total Inference Runs per Model: {len(test_images) * NUM_INFERENCE_RUNS}\n")
        f.write(f"- Total Execution Time: {total_time:.2f}s\n\n")
        
        # Results table
        f.write(f"## Results\n\n")
        f.write(f"| Model | Load Time (s) | Avg Inference (s) | Std (s) | FPS | Peak Memory (MB) | Parameters | Layers |\n")
        f.write(f"|-------|---------------|-------------------|---------|-----|------------------|------------|--------|\n")
        for r in results:
            f.write(f"| {r['model_name']} | {r['load_time_sec']:.2f} | "
                   f"{r['avg_inference_sec']:.3f} | {r['std_inference_sec']:.3f} | "
                   f"{r['fps_avg']:.2f} | {r['mem_peak_mb']:.0f} | "
                   f"{r['total_params']:,} | {r['total_layers_dissected']} |\n")
        
        # Analysis
        f.write(f"\n## Analysis\n\n")
        
        fastest = min(results, key=lambda x: x['avg_inference_sec'])
        slowest = max(results, key=lambda x: x['avg_inference_sec'])
        most_memory = max(results, key=lambda x: x['mem_peak_mb'])
        most_params = max(results, key=lambda x: x['total_params'])
        
        f.write(f"- **Fastest Model:** {fastest['model_name']} ({fastest['avg_inference_sec']:.3f}s, {fastest['fps_avg']:.2f} FPS)\n")
        f.write(f"- **Slowest Model:** {slowest['model_name']} ({slowest['avg_inference_sec']:.3f}s, {slowest['fps_avg']:.2f} FPS)\n")
        f.write(f"- **Most Memory Intensive:** {most_memory['model_name']} ({most_memory['mem_peak_mb']:.0f} MB)\n")
        f.write(f"- **Most Parameters:** {most_params['model_name']} ({most_params['total_params']:,})\n\n")
        
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
    print(f"Total Inference Runs: {len(test_images) * NUM_INFERENCE_RUNS * len(results)}")
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
```

## Key Implementation Notes

### Inline Comments Strategy
Every model class should include:
1. Config explanation comments above each constant
2. Dependency requirements with setup instructions
3. Auto-download behavior notes
4. Expected input/output format comments
5. Model-specific preprocessing notes

### Memory Management
1. Use `torch.cuda.synchronize()` before timing measurements
2. Call `torch.cuda.empty_cache()` after each layer dissection
3. Explicitly `del` large tensors immediately after use
4. Run `gc.collect()` during cleanup phase
5. Track memory at key checkpoints: before load, after load, peak inference, after cleanup

### Error Handling
1. Wrap entire model processing in try/except
2. Print full traceback with `traceback.print_exc()`
3. Exit immediately with `sys.exit(1)` on any error
4. No partial result saving - fail-fast philosophy

### Progress Logging Format
```
================================================================================
[1/8] DepthProBenchmark
================================================================================
[CHECK] Verifying model weights and dependencies...
[MEMORY] GPU Memory before load: 512 MB
[LOAD] Loading model to cuda...
[LOAD] Model loaded successfully in 5.23s
[MEMORY] GPU Memory after load: 2560 MB (delta: +2048 MB)
[INFER 1/15] test_image.jpg | Time: 2.15s | Peak Memory: 3072 MB
[INFER 2/15] test_image.jpg | Time: 2.10s | Peak Memory: 3072 MB
...
[STATS] Inference: avg=2.122s std=0.018s min=2.10s max=2.15s
[STATS] FPS: avg=0.471 std=0.004
[DISSECT] Starting layer dissection (245 layers detected)...
[DISSECT 1/245] Layer 001: torch.Size([1, 64, 320, 320])
[DISSECT 2/245] Layer 002: torch.Size([1, 64, 320, 320])
...
[DISSECT] Saved 245 layer visualizations to viz/20251110_143022/DepthProBenchmark/
[CLEANUP] Clearing model and GPU cache...
[MEMORY] GPU Memory after cleanup: 512 MB
[COMPLETE] DepthProBenchmark finished
================================================================================
```

## File Structure

```
vision-bench/
├── unified_benchmark.py          # Single file containing all code
├── results/
│   ├── benchmark_results_20251110_143022.json
│   ├── benchmark_summary_20251110_143022.csv
│   └── benchmark_report_20251110_143022.md
└── viz/
    └── 20251110_143022/
        ├── DepthProBenchmark/
        │   ├── layer_001.png
        │   ├── layer_001.npy
        │   ├── layer_002.png
        │   ├── layer_002.npy
        │   └── layers_metadata.json
        ├── DepthAnythingV2SmallBenchmark/
        ├── DepthAnythingV2BaseBenchmark/
        ├── DepthAnythingV2LargeBenchmark/
        ├── YOLO11nDetectBenchmark/
        ├── YOLO11nSegmentBenchmark/
        ├── YOLO11nPoseBenchmark/
        └── MobileSAMBenchmark/
```

## Adding New Models (Developer Guide)

To add a new model to the benchmark:

1. Create a new class inheriting from `ModelBenchmark`:
```python
class YourModelBenchmark(ModelBenchmark):
    # Configuration - modify these for your model
    MODEL_NAME = "YourModel"
    MODEL_PATH = "../models/your_model.pt"  # Or model ID
    INPUT_SIZE = 512  # Your model's native input size
    
    # Add comment explaining model purpose and requirements
    # Example: Uses custom library, requires xyz package installed
    
    def __init__(self, device):
        self.device = device
        self.model = None
    
    def load(self):
        # Load your model here
        # Handle auto-download if needed
        # Set model to eval mode
        pass
    
    def infer(self, image_path: Path) -> float:
        # Preprocess image according to model requirements
        # Time the inference with CUDA sync if GPU
        # Return elapsed time in seconds
        pass
    
    def dissect_layers(self, image_path: Path) -> dict:
        # Follow the memory-efficient pattern:
        # For each layer: hook → infer → capture → save → delete
        # Return metadata dictionary
        pass
    
    def cleanup(self):
        # Delete model, clear cache, collect garbage
        pass
    
    def get_info(self) -> dict:
        # Return model metadata (params, layers, etc)
        pass
```

2. Add your class to the `benchmark_classes` list in `run_benchmark()`

3. Run the benchmark - your model will be processed sequentially with all others

## Resolved Considerations

1. **Image preprocessing**: Handled per-model in `infer()` method
2. **Output tensor handling**: Each model's `infer()` handles its specific output format, only returns timing
3. **Layer dissection images**: Use first test image only for consistency
4. **CSV columns**: Ordered by importance (model_name, timings, FPS, memory, params)
5. **Markdown precision**: Round to 2-3 decimals for readability (.2f for load time, .3f for inference)
