"use client";

import { useState, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";

interface LayerData {
  model: string;
  layers: {
    idx: number;
    name: string;
    type: string;
    shape: number[];
    min: number;
    max: number;
    mean: number;
    std: number;
    sparsity: number;
  }[];
  visualizations: string[];
}

export function LayerExplorer() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [layerData, setLayerData] = useState<LayerData | null>(null);
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingLayers, setLoadingLayers] = useState(false);
  const [imageLoading, setImageLoading] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadLayerData(selectedModel);
    }
  }, [selectedModel]);

  const loadModels = async () => {
    try {
      console.log("[Layer Explorer] Fetching models from /api/models");
      const response = await fetch("/api/models");
      console.log("[Layer Explorer] Models response status:", response.status);

      if (!response.ok) {
        console.error(
          "[Layer Explorer] Models fetch failed with status:",
          response.status
        );
        setLoading(false);
        return;
      }

      const data = await response.json();
      console.log("[Layer Explorer] Received models:", data.models);
      setModels(data.models || []);

      if (data.models && data.models.length > 0) {
        console.log(
          "[Layer Explorer] Auto-selecting first model:",
          data.models[0]
        );
        setSelectedModel(data.models[0]);
      } else {
        console.warn("[Layer Explorer] No models found in response");
      }
    } catch (error) {
      console.error("[Layer Explorer] Error loading models:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadLayerData = async (model: string) => {
    setLoadingLayers(true);
    try {
      console.log(`[Layer Explorer] Fetching layer data for model: ${model}`);
      const response = await fetch(`/api/layers/${model}`);
      console.log(
        "[Layer Explorer] Layer data response status:",
        response.status
      );

      if (!response.ok) {
        console.error(
          "[Layer Explorer] Layer data fetch failed with status:",
          response.status
        );
        return;
      }

      const data = await response.json();
      console.log(`[Layer Explorer] Received layer data for ${model}:`, {
        layersCount: data.layers?.length || 0,
        visualizationsCount: data.visualizations?.length || 0,
      });

      if (data.layers && data.layers.length > 0) {
        console.log("[Layer Explorer] First layer:", data.layers[0]);
      } else {
        console.warn("[Layer Explorer] No layers found in response");
      }

      if (data.visualizations && data.visualizations.length > 0) {
        console.log(
          "[Layer Explorer] First visualization path:",
          data.visualizations[0]
        );
      } else {
        console.warn("[Layer Explorer] No visualizations found in response");
      }

      setLayerData(data);
      if (data.layers && data.layers.length > 0) {
        console.log("[Layer Explorer] Auto-selecting first layer");
        setSelectedLayer(0);
      }
    } catch (error) {
      console.error("[Layer Explorer] Error loading layer data:", error);
    } finally {
      setLoadingLayers(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <Spinner size="lg" />
        <p className="text-muted-foreground">Loading models...</p>
      </div>
    );
  }

  const currentLayer =
    layerData && selectedLayer !== null
      ? layerData.layers[selectedLayer]
      : null;

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-2 border rounded-md">
        <div className="p-4 border-b">
          <h3 className="font-semibold text-sm">Models</h3>
        </div>
        <ScrollArea className="h-[600px]">
          <div className="p-2">
            {models.map((model) => (
              <button
                key={model}
                onClick={() => setSelectedModel(model)}
                className={`w-full text-left p-2 rounded border mb-2 text-sm transition-colors ${
                  selectedModel === model
                    ? "bg-secondary border-primary"
                    : "border-border hover:border-primary"
                }`}
              >
                {model}
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      <div className="col-span-3 border rounded-md">
        <div className="p-4 border-b">
          <h3 className="font-semibold text-sm">Layers</h3>
          {layerData && (
            <p className="text-xs text-muted-foreground mt-1">
              {layerData.layers.length} layers
            </p>
          )}
        </div>
        <ScrollArea className="h-[600px]">
          <div className="p-2">
            {loadingLayers ? (
              <div className="flex flex-col items-center justify-center py-16 space-y-4">
                <Spinner />
                <p className="text-xs text-muted-foreground">
                  Loading layers...
                </p>
              </div>
            ) : (
              layerData?.layers.map((layer, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setSelectedLayer(idx);
                    setImageLoading(true);
                  }}
                  className={`w-full text-left p-2 rounded border mb-2 text-sm transition-colors ${
                    selectedLayer === idx
                      ? "bg-secondary border-primary"
                      : "border-border hover:border-primary"
                  }`}
                >
                  <div className="font-medium">{layer.name}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {layer.type} | {layer.shape.join("x")}
                  </div>
                </button>
              ))
            )}
          </div>
        </ScrollArea>
      </div>

      <div className="col-span-7">
        <div className="border rounded-md">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Layer Details</h3>
          </div>
          <ScrollArea className="h-[600px]">
            {currentLayer ? (
              <div className="p-4 space-y-4">
                <div>
                  <h4 className="font-semibold mb-2">{currentLayer.name}</h4>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Type</span>
                        <span className="font-mono">{currentLayer.type}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Shape</span>
                        <span className="font-mono">
                          [{currentLayer.shape.join(", ")}]
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Min</span>
                        <span className="font-mono">
                          {currentLayer.min !== null
                            ? currentLayer.min.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Max</span>
                        <span className="font-mono">
                          {currentLayer.max !== null
                            ? currentLayer.max.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Mean</span>
                        <span className="font-mono">
                          {currentLayer.mean !== null
                            ? currentLayer.mean.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Std Dev</span>
                        <span className="font-mono">
                          {currentLayer.std !== null
                            ? currentLayer.std.toFixed(4)
                            : "N/A"}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-muted-foreground">Sparsity</span>
                        <span className="font-mono">
                          {currentLayer.sparsity !== null
                            ? (currentLayer.sparsity * 100).toFixed(2) + "%"
                            : "N/A"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="border-t pt-4">
                  <h4 className="font-semibold mb-3">Visualization</h4>
                  {selectedLayer !== null &&
                  layerData?.visualizations[selectedLayer] ? (
                    <div className="border rounded p-2 relative">
                      {imageLoading && (
                        <div className="absolute inset-0 flex items-center justify-center bg-background/80 z-10 rounded">
                          <Spinner size="lg" />
                        </div>
                      )}
                      <img
                        src={
                          layerData.visualizations[selectedLayer].startsWith(
                            "https://"
                          )
                            ? layerData.visualizations[selectedLayer]
                            : `/api/viz${layerData.visualizations[selectedLayer]}`
                        }
                        alt={`Layer ${selectedLayer} visualization`}
                        className="w-full h-auto"
                        onLoadStart={() => setImageLoading(true)}
                        onLoad={() => setImageLoading(false)}
                        onError={() => setImageLoading(false)}
                      />
                    </div>
                  ) : (
                    <p className="text-sm text-muted-foreground">
                      No visualization available
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="p-8 text-center text-muted-foreground">
                Select a layer to view details
              </div>
            )}
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
