import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const cwd = process.cwd();
    const vizPath = path.join(cwd, "..", "vision-bench", "viz");
    
    console.log("[Benchmarks API] Current working directory:", cwd);
    console.log("[Benchmarks API] Looking for viz path:", vizPath);
    console.log("[Benchmarks API] Viz path exists:", fs.existsSync(vizPath));

    if (!fs.existsSync(vizPath)) {
      console.warn("[Benchmarks API] Viz path does not exist, returning empty runs");
      return NextResponse.json({ runs: [] });
    }

    const allFiles = fs.readdirSync(vizPath);
    console.log("[Benchmarks API] All files/dirs in viz:", allFiles);
    
    const runs = fs
      .readdirSync(vizPath)
      .filter((file) => {
        const isDir = fs.statSync(path.join(vizPath, file)).isDirectory();
        console.log(`[Benchmarks API] ${file} is directory: ${isDir}`);
        return isDir;
      })
      .map((timestamp) => {
        const runPath = path.join(vizPath, timestamp);
        const modelDirs = fs.readdirSync(runPath);
        console.log(`[Benchmarks API] Run ${timestamp} contains:`, modelDirs);
        
        const models = modelDirs
          .filter((file) => {
            const modelPath = path.join(runPath, file);
            const isDir = fs.statSync(modelPath).isDirectory();
            console.log(`[Benchmarks API]   ${file} is directory: ${isDir}`);
            return isDir;
          });
        
        console.log(`[Benchmarks API] Run ${timestamp} models:`, models);

        return {
          timestamp,
          models,
          path: runPath,
        };
      })
      .sort((a, b) => b.timestamp.localeCompare(a.timestamp));

    console.log("[Benchmarks API] Total runs found:", runs.length);
    console.log("[Benchmarks API] Returning runs:", runs.map(r => ({ timestamp: r.timestamp, modelCount: r.models.length })));
    return NextResponse.json({ runs });
  } catch (error) {
    console.error("[Benchmarks API] Error reading benchmarks:", error);
    console.error("[Benchmarks API] Error stack:", error instanceof Error ? error.stack : 'N/A');
    return NextResponse.json({ runs: [] });
  }
}
