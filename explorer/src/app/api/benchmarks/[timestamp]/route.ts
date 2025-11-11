import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(
  request: Request,
  { params }: { params: { timestamp: string } }
) {
  try {
    const cwd = process.cwd();
    const resultsPath = path.join(
      cwd,
      "..",
      "vision-bench",
      "results",
      `benchmark_results_${params.timestamp}.json`
    );

    console.log(
      `[Benchmark Metrics API] Request for timestamp: ${params.timestamp}`
    );
    console.log("[Benchmark Metrics API] Current working directory:", cwd);
    console.log("[Benchmark Metrics API] Results path:", resultsPath);
    console.log(
      "[Benchmark Metrics API] Results file exists:",
      fs.existsSync(resultsPath)
    );

    if (!fs.existsSync(resultsPath)) {
      console.warn(
        `[Benchmark Metrics API] Results file not found: ${resultsPath}`
      );

      // Check what files actually exist in the results directory
      const resultsDir = path.join(cwd, "..", "vision-bench", "results");
      if (fs.existsSync(resultsDir)) {
        const availableFiles = fs.readdirSync(resultsDir);
        console.log(
          "[Benchmark Metrics API] Available files in results directory:",
          availableFiles
        );
      } else {
        console.warn(
          "[Benchmark Metrics API] Results directory does not exist:",
          resultsDir
        );
      }

      return NextResponse.json({ metrics: [] });
    }

    const fileContent = fs.readFileSync(resultsPath, "utf-8");
    console.log(
      `[Benchmark Metrics API] File size: ${fileContent.length} bytes`
    );

    const data = JSON.parse(fileContent);
    console.log("[Benchmark Metrics API] Parsed data keys:", Object.keys(data));

    // Check if data has 'results' array (new format) or direct model entries (old format)
    let resultsArray;
    if (Array.isArray(data.results)) {
      console.log(
        "[Benchmark Metrics API] Using new format with 'results' array"
      );
      resultsArray = data.results;
    } else {
      console.log(
        "[Benchmark Metrics API] Using old format with direct model entries"
      );
      resultsArray = Object.entries(data).map(([model, stats]) => ({
        model_name: model,
        ...(stats as Record<string, any>),
      }));
    }

    console.log(
      `[Benchmark Metrics API] Found ${resultsArray.length} model results`
    );

    const metrics = resultsArray.map((result: any) => {
      const metric = {
        model: result.model_name || result.model || "Unknown",
        loadTime: result.load_time_sec || result.load_time || 0,
        avgInference:
          result.avg_inference_sec || result.avg_inference_time || 0,
        std: result.std_inference_sec || result.std || 0,
        fps: result.fps_avg || result.fps || 0,
        peakMemory: result.mem_peak_mb || result.peak_memory_mb || 0,
        parameters: result.total_params || result.parameters || 0,
        layers: result.total_layers_dissected || result.layers || 0,
      };
      console.log(
        `[Benchmark Metrics API] Processed metric for ${metric.model}:`,
        metric
      );
      return metric;
    });

    console.log(`[Benchmark Metrics API] Returning ${metrics.length} metrics`);
    return NextResponse.json({ metrics });
  } catch (error) {
    console.error(
      `[Benchmark Metrics API] Error reading benchmark metrics for timestamp ${params.timestamp}:`,
      error
    );
    console.error(
      "[Benchmark Metrics API] Error stack:",
      error instanceof Error ? error.stack : "N/A"
    );
    return NextResponse.json({ metrics: [] });
  }
}
