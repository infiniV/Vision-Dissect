# Vision-Dissect Explorer

A professional Next.js web application for exploring vision model benchmarks, layer activations, and computational graphs. Built with React, TypeScript, and shadcn/ui components.

## Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

### Data Source Configuration

The explorer automatically switches between two data sources:

- **Development** (local): Reads from `../vision-bench/` directory (filesystem)
- **Production** (Vercel): Fetches from GitHub repository raw content

**Optional**: Add a GitHub token for higher API rate limits in production:

1. Create a `.env.local` file in the explorer directory
2. Add: `GITHUB_TOKEN=ghp_yourTokenHere`
3. Get a token at: https://github.com/settings/tokens (no special permissions needed)

## Architecture

### Technology Stack

- **Framework**: Next.js 14 (App Router)
- **UI Components**: shadcn/ui (Radix UI primitives)
- **Styling**: Tailwind CSS
- **Graph Visualization**: @xyflow/react (ReactFlow)
- **Charts**: Recharts
- **Icons**: Lucide React

### Project Structure

```
explorer/
├── src/
│   ├── app/
│   │   ├── api/                          # Backend API routes
│   │   │   ├── benchmarks/
│   │   │   │   ├── route.ts              # List all benchmark runs
│   │   │   │   └── [timestamp]/route.ts  # Get metrics for specific run
│   │   │   ├── models/route.ts           # List available models
│   │   │   ├── layers/[model]/route.ts   # Get layer data for model
│   │   │   ├── status/route.ts           # Get live benchmark status
│   │   │   └── viz/[...path]/route.ts    # Serve visualization images
│   │   ├── page.tsx                      # Main application page
│   │   ├── layout.tsx                    # Root layout
│   │   └── globals.css                   # Global styles
│   ├── components/
│   │   ├── ui/                           # shadcn/ui base components
│   │   ├── benchmark-explorer.tsx        # Benchmarks tab component
│   │   ├── layer-explorer.tsx            # Layer visualizations tab
│   │   ├── layer-graph-view.tsx          # Graph view tab
│   │   ├── layer-node.tsx                # Custom ReactFlow node
│   │   └── metrics-monitor.tsx           # Live monitor tab
│   └── lib/
│       └── utils.ts                      # Utility functions
├── package.json
├── next.config.mjs
├── tailwind.config.ts
└── tsconfig.json
```

## Features

### 1. Benchmarks Tab

View and compare performance metrics across benchmark runs:

- **Benchmark Run List**: Browse all benchmark runs by timestamp
- **Model Metrics**: Expandable accordion showing detailed metrics per model
  - Load time and inference time (avg, std dev)
  - FPS (frames per second)
  - Peak memory usage (MB)
  - Model parameters (millions)
  - Number of layers

**Data Source**: Reads from `vision-bench/results/benchmark_results_{timestamp}.json`

### 2. Layer Visualizations Tab

Explore individual layers within models:

- **Model Selector**: Browse available models from latest benchmark run
- **Layer List**: View all layers with name, type, and shape
- **Layer Details**: Display comprehensive layer statistics
  - Shape, type, index
  - Statistical measures (min, max, mean, std dev, sparsity)
  - High-resolution PNG visualization of layer activations

**Data Sources**:

- Layer metadata from `vision-bench/viz/{timestamp}/{model}/layers_metadata.json`
- Visualizations from `vision-bench/viz/{timestamp}/{model}/*.png`

### 3. Graph View Tab

Interactive computational graph visualization:

- **Flow Diagram**: Nodes represent layers, edges show data flow
- **Layer Nodes**: Each node displays:
  - Layer thumbnail visualization
  - Layer index, name, and type
  - Shape and key statistics (mean, std, sparsity)
- **Interactive Controls**:
  - Zoom, pan, minimap navigation
  - Node selection and highlighting
  - Animated edges showing forward pass

**Visualization**: Uses ReactFlow for graph rendering with custom layer node components

### 4. Live Monitor Tab

Real-time monitoring of running benchmark processes:

- **Status Cards**: Show running state, current model, progress, start time
- **Process Logs**: Live log output from benchmark scripts
- **Auto-refresh**: Polls status every 2 seconds

**Data Source**: Reads from `vision-bench/.status.json` (optional, created by benchmark scripts)

## API Routes

### `/api/benchmarks`

- **Method**: GET
- **Returns**: List of all benchmark runs with timestamps and model names
- **Source**: Scans `vision-bench/viz/` for timestamped directories

### `/api/benchmarks/[timestamp]`

- **Method**: GET
- **Params**: `timestamp` - Benchmark run timestamp
- **Returns**: Performance metrics for all models in the run
- **Source**: `vision-bench/results/benchmark_results_{timestamp}.json`

### `/api/models`

- **Method**: GET
- **Returns**: List of model names from the latest benchmark run
- **Source**: Subdirectories in `vision-bench/viz/{latest_timestamp}/`

### `/api/layers/[model]`

- **Method**: GET
- **Params**: `model` - Model name
- **Returns**: Layer metadata and visualization paths for the model
- **Source**:
  - `vision-bench/viz/{latest_timestamp}/{model}/layers_metadata.json`
  - PNG files in the same directory

### `/api/status`

- **Method**: GET
- **Returns**: Current benchmark process status and logs
- **Source**: `vision-bench/.status.json` (optional)

### `/api/viz/[...path]`

- **Method**: GET
- **Params**: `path` - Relative path to visualization file
- **Returns**: Image file (PNG/JPEG)
- **Source**:
  - Development: Serves files from `vision-bench/viz/{path}`
  - Production: Proxies from GitHub raw content

## Data Flow

1. **Benchmark Scripts** (Python) generate data in `vision-bench/`:

   - `viz/{timestamp}/{model}/` - Layer visualizations and metadata
   - `results/benchmark_results_{timestamp}.json` - Performance metrics
   - `.status.json` - Real-time status (optional)

2. **Next.js API Routes** read data from:

   - **Development**: Local filesystem (`../vision-bench/`)
   - **Production**: GitHub raw content (https://raw.githubusercontent.com/infiniV/Vision-Dissect/main/vision-bench/)

3. **React Components** fetch from API routes and render UI

4. **Data Refresh**:
   - Development: Real-time (reads latest files)
   - Production: Cached for 60 seconds (Next.js revalidation)

## Design Philosophy

- **Minimalist**: Black and white color scheme, professional research tool aesthetic
- **No Emojis**: Text-only UI elements for scientific presentation
- **Responsive**: Grid-based layouts adapt to screen sizes
- **Real-time**: Auto-refreshing monitor tab for live feedback
- **Accessible**: Using Radix UI primitives ensures ARIA compliance

## Data Format Examples

### Benchmark Results JSON

```json
{
  "results": [
    {
      "model_name": "yolo11n",
      "load_time_sec": 0.123,
      "avg_inference_sec": 0.045,
      "std_inference_sec": 0.002,
      "fps_avg": 22.2,
      "mem_peak_mb": 256,
      "total_params": 2500000,
      "total_layers_dissected": 45
    }
  ]
}
```

### Layer Metadata JSON

```json
{
  "layers": [
    {
      "idx": 0,
      "name": "conv1",
      "type": "Conv2d",
      "shape": [1, 64, 112, 112],
      "min": -0.5234,
      "max": 1.2345,
      "mean": 0.0123,
      "std": 0.2345,
      "sparsity": 0.15
    }
  ]
}
```

### Status JSON

```json
{
  "running": true,
  "currentModel": "yolo11n",
  "progress": 2,
  "totalModels": 5,
  "startTime": "2025-11-11T10:30:00",
  "logs": ["Starting benchmark...", "Processing yolo11n..."]
}
```

## Development

### Running Tests

```bash
npm run test        # Run Playwright tests
npm run test:ui     # Run tests with UI
```

### Building for Production

```bash
npm run build       # Create production build
npm start           # Start production server
```

### Linting

```bash
npm run lint        # Run ESLint
## Deployment

### Vercel Deployment

The app is production-ready for Vercel deployment:

1. Push changes to GitHub repository
2. Import project in Vercel dashboard
3. No build configuration needed (auto-detected)
4. Optional: Add `GITHUB_TOKEN` environment variable in Vercel settings

The app automatically detects the production environment and switches to GitHub data source.

### GitHub Repository Requirements

Ensure your repository includes:
- `vision-bench/viz/` - Layer visualizations (PNG files only, NPY excluded)
- `vision-bench/results/` - Benchmark result JSONs
- `.gitignore` properly excludes `**/*.npy` files

## Notes

- **Development**: Requires `vision-bench/` directory as sibling of `explorer/`
- **Production**: Fetches all data from GitHub (no local files needed)
- All file system operations use Node.js `fs` module (server-side only)
- Image optimization is handled by Next.js Image component
- Console logging is extensive for debugging API routes
- GitHub API has rate limits: 60 requests/hour (no token) or 5000 requests/hour (with token)erver-side only)
- Image optimization is handled by Next.js Image component
- Console logging is extensive for debugging API routes
```
