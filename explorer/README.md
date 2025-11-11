# Vision-Dissect Explorer

Simple, professional React frontend for exploring model benchmarks and layer visualizations.

## Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Features

### Benchmarks Tab

- View all benchmark runs
- Compare performance metrics across models
- Inspect inference times, FPS, memory usage, parameters

### Layer Visualizations Tab

- Browse models and their layers
- View layer metadata (shape, statistics, sparsity)
- Display PNG visualizations for each layer

### Live Monitor Tab

- Monitor running benchmark scripts
- View real-time progress
- Display process logs

## Design

- Black and white minimalist design
- shadcn/ui components
- No emojis, professional research tool
- Simple navigation with tabs

## Directory Structure

```
explorer/
├── src/
│   ├── app/
│   │   ├── api/          # API routes for data fetching
│   │   ├── page.tsx      # Main page
│   │   └── globals.css   # Global styles
│   ├── components/
│   │   ├── ui/           # shadcn/ui components
│   │   ├── benchmark-explorer.tsx
│   │   ├── layer-explorer.tsx
│   │   └── metrics-monitor.tsx
│   └── lib/
│       └── utils.ts      # Utilities
├── package.json
└── README.md
```

## Data Sources

The explorer reads from:

- `vision-bench/viz/` - Visualization outputs
- `vision-bench/results/` - Benchmark results
- `vision-bench/.status.json` - Live process status (optional)
