import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(
  request: Request,
  { params }: { params: { path: string[] } }
) {
  try {
    const cwd = process.cwd();
    const filePath = path.join(
      cwd,
      "..",
      "vision-bench",
      "viz",
      ...params.path
    );

    console.log(`[Viz API] Request for file path:`, params.path);
    console.log("[Viz API] Current working directory:", cwd);
    console.log("[Viz API] Full file path:", filePath);
    console.log("[Viz API] File exists:", fs.existsSync(filePath));

    if (!fs.existsSync(filePath)) {
      console.warn(`[Viz API] File not found: ${filePath}`);
      
      // Try to list parent directory to help debug
      const parentDir = path.dirname(filePath);
      if (fs.existsSync(parentDir)) {
        const parentFiles = fs.readdirSync(parentDir);
        console.log(`[Viz API] Files in parent directory (${parentDir}):`, parentFiles);
      } else {
        console.warn(`[Viz API] Parent directory does not exist: ${parentDir}`);
      }
      
      return new NextResponse("Not found", { status: 404 });
    }

    const stats = fs.statSync(filePath);
    console.log(`[Viz API] File size: ${stats.size} bytes`);
    
    const file = fs.readFileSync(filePath);
    const ext = path.extname(filePath).toLowerCase();
    console.log(`[Viz API] File extension: ${ext}`);

    const contentTypes: Record<string, string> = {
      ".png": "image/png",
      ".jpg": "image/jpeg",
      ".jpeg": "image/jpeg",
    };

    const contentType = contentTypes[ext] || "application/octet-stream";
    console.log(`[Viz API] Content-Type: ${contentType}`);
    console.log(`[Viz API] Successfully serving file`);
    
    return new NextResponse(file, {
      headers: {
        "Content-Type": contentType,
      },
    });
  } catch (error) {
    console.error("[Viz API] Error serving file:", error);
    console.error("[Viz API] Error stack:", error instanceof Error ? error.stack : 'N/A');
    console.error("[Viz API] Request path:", params.path);
    return new NextResponse("Error", { status: 500 });
  }
}
