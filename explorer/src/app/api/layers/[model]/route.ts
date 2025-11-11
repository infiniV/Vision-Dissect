import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(
  request: Request,
  { params }: { params: { model: string } }
) {
  try {
    const cwd = process.cwd();
    const vizPath = path.join(cwd, "..", "vision-bench", "viz");

    console.log(`[Layers API] Request for model: ${params.model}`);
    console.log("[Layers API] Current working directory:", cwd);
    console.log("[Layers API] Viz path:", vizPath);

    const runs = fs
      .readdirSync(vizPath)
      .filter((file) => fs.statSync(path.join(vizPath, file)).isDirectory())
      .sort((a, b) => b.localeCompare(a));

    console.log("[Layers API] Found runs:", runs);

    if (runs.length === 0) {
      console.warn("[Layers API] No runs found");
      return NextResponse.json({ layers: [], visualizations: [] });
    }

    const latestRun = runs[0];
    const modelPath = path.join(vizPath, latestRun, params.model);

    console.log("[Layers API] Latest run:", latestRun);
    console.log("[Layers API] Model path:", modelPath);
    console.log("[Layers API] Model path exists:", fs.existsSync(modelPath));

    if (!fs.existsSync(modelPath)) {
      console.warn(`[Layers API] Model path does not exist: ${modelPath}`);
      return NextResponse.json({ layers: [], visualizations: [] });
    }

    const allFiles = fs.readdirSync(modelPath);
    console.log(`[Layers API] All files in model directory:`, allFiles);

    const metadataPath = path.join(modelPath, "layers_metadata.json");
    console.log("[Layers API] Metadata path:", metadataPath);
    console.log("[Layers API] Metadata exists:", fs.existsSync(metadataPath));

    let layers = [];
    if (fs.existsSync(metadataPath)) {
      const metadataContent = fs.readFileSync(metadataPath, "utf-8");
      console.log(
        `[Layers API] Metadata file size: ${metadataContent.length} bytes`
      );
      const metadata = JSON.parse(metadataContent);
      layers = metadata.layers || [];
      console.log(`[Layers API] Loaded ${layers.length} layers from metadata`);
      if (layers.length > 0) {
        console.log(`[Layers API] First layer:`, layers[0]);
        console.log(`[Layers API] Last layer:`, layers[layers.length - 1]);
      }
    } else {
      console.warn("[Layers API] No metadata file found");
    }

    const pngFiles = fs
      .readdirSync(modelPath)
      .filter((file) => {
        const isPng = file.endsWith(".png");
        if (isPng) console.log(`[Layers API] Found PNG: ${file}`);
        return isPng;
      })
      .sort((a, b) => {
        const aNum = parseInt(a.match(/\d+/)?.[0] || "0");
        const bNum = parseInt(b.match(/\d+/)?.[0] || "0");
        return aNum - bNum;
      });

    console.log(`[Layers API] Found ${pngFiles.length} PNG files`);

    const visualizations = pngFiles.map(
      (file) => `/${latestRun}/${params.model}/${file}`
    );

    console.log(
      `[Layers API] Visualization paths:`,
      visualizations.slice(0, 3),
      visualizations.length > 3
        ? `... and ${visualizations.length - 3} more`
        : ""
    );

    const response = {
      model: params.model,
      layers,
      visualizations,
    };

    console.log(
      `[Layers API] Returning response with ${layers.length} layers and ${visualizations.length} visualizations`
    );
    return NextResponse.json(response);
  } catch (error) {
    console.error(
      `[Layers API] Error reading layer data for model ${params.model}:`,
      error
    );
    console.error(
      "[Layers API] Error stack:",
      error instanceof Error ? error.stack : "N/A"
    );
    return NextResponse.json({ layers: [], visualizations: [] });
  }
}
