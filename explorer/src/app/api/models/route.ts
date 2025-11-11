import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const cwd = process.cwd();
    const vizPath = path.join(cwd, "..", "vision-bench", "viz");
    
    console.log("[Models API] Current working directory:", cwd);
    console.log("[Models API] Looking for viz path:", vizPath);
    console.log("[Models API] Viz path exists:", fs.existsSync(vizPath));

    if (!fs.existsSync(vizPath)) {
      console.warn("[Models API] Viz path does not exist, returning empty models");
      return NextResponse.json({ models: [] });
    }

    const allFiles = fs.readdirSync(vizPath);
    console.log("[Models API] All files/dirs in viz:", allFiles);
    
    const runs = fs
      .readdirSync(vizPath)
      .filter((file) => {
        const isDir = fs.statSync(path.join(vizPath, file)).isDirectory();
        console.log(`[Models API] ${file} is directory: ${isDir}`);
        return isDir;
      })
      .sort((a, b) => b.localeCompare(a));

    console.log("[Models API] Found runs:", runs);
    
    if (runs.length === 0) {
      console.warn("[Models API] No runs found, returning empty models");
      return NextResponse.json({ models: [] });
    }

    const latestRun = runs[0];
    const latestRunPath = path.join(vizPath, latestRun);
    console.log("[Models API] Latest run:", latestRun);
    console.log("[Models API] Latest run path:", latestRunPath);
    
    const allModelDirs = fs.readdirSync(latestRunPath);
    console.log("[Models API] All items in latest run:", allModelDirs);
    
    const models = allModelDirs
      .filter((file) => {
        const modelPath = path.join(latestRunPath, file);
        const isDir = fs.statSync(modelPath).isDirectory();
        console.log(`[Models API] ${file} is directory: ${isDir}`);
        return isDir;
      });

    console.log("[Models API] Found models:", models);
    return NextResponse.json({ models });
  } catch (error) {
    console.error("[Models API] Error reading models:", error);
    console.error("[Models API] Error stack:", error instanceof Error ? error.stack : 'N/A');
    return NextResponse.json({ models: [] });
  }
}
