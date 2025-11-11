import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET() {
  try {
    const cwd = process.cwd();
    const statusPath = path.join(
      cwd,
      "..",
      "vision-bench",
      ".status.json"
    );

    console.log("[Status API] Current working directory:", cwd);
    console.log("[Status API] Status file path:", statusPath);
    console.log("[Status API] Status file exists:", fs.existsSync(statusPath));

    if (!fs.existsSync(statusPath)) {
      console.log("[Status API] No status file found, returning idle state");
      return NextResponse.json({
        running: false,
        currentModel: null,
        progress: 0,
        totalModels: 0,
        startTime: null,
        logs: [],
      });
    }

    const statusContent = fs.readFileSync(statusPath, "utf-8");
    console.log(`[Status API] Status file size: ${statusContent.length} bytes`);
    
    const status = JSON.parse(statusContent);
    console.log("[Status API] Status:", {
      running: status.running,
      currentModel: status.currentModel,
      progress: status.progress,
      totalModels: status.totalModels,
      logsCount: status.logs?.length || 0
    });
    
    return NextResponse.json(status);
  } catch (error) {
    console.error("[Status API] Error reading status:", error);
    console.error("[Status API] Error stack:", error instanceof Error ? error.stack : 'N/A');
    return NextResponse.json({
      running: false,
      currentModel: null,
      progress: 0,
      totalModels: 0,
      startTime: null,
      logs: [],
    });
  }
}
