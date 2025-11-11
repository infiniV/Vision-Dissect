"use client";

import { useState, useEffect, useCallback, useMemo } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  applyNodeChanges,
  applyEdgeChanges,
  NodeChange,
  EdgeChange,
  ConnectionMode,
} from "@xyflow/react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Spinner } from "@/components/ui/spinner";
import LayerNode, { LayerNodeData } from "@/components/layer-node";

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

export function LayerGraphView() {
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [layerData, setLayerData] = useState<LayerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingGraph, setLoadingGraph] = useState(false);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  // Define node types
  const nodeTypes = useMemo(() => ({ layerNode: LayerNode }), []);

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (selectedModel) {
      loadLayerData(selectedModel);
    }
  }, [selectedModel]);

  // Convert layer data to nodes and edges
  useEffect(() => {
    if (layerData) {
      const newNodes: Node[] = [];
      const newEdges: Edge[] = [];

      // Calculate vertical spacing
      const nodeHeight = 250; // Approximate height of each node
      const verticalGap = 50;
      const startY = 50;

      layerData.layers.forEach((layer, index) => {
        const nodeData: LayerNodeData = {
          idx: layer.idx,
          name: layer.name,
          type: layer.type,
          shape: layer.shape,
          min: layer.min,
          max: layer.max,
          mean: layer.mean,
          std: layer.std,
          sparsity: layer.sparsity,
          visualizationPath: layerData.visualizations[index] || "",
        };

        // Create node
        newNodes.push({
          id: `layer-${layer.idx}`,
          type: "layerNode",
          position: {
            x: 400, // Center horizontally
            y: startY + index * (nodeHeight + verticalGap),
          },
          data: nodeData,
        });

        // Create edge connecting to previous layer
        if (index > 0) {
          newEdges.push({
            id: `edge-${layer.idx - 1}-${layer.idx}`,
            source: `layer-${layerData.layers[index - 1].idx}`,
            target: `layer-${layer.idx}`,
            type: "smoothstep",
            animated: true,
            style: { stroke: "#666", strokeWidth: 2 },
          });
        }
      });

      setNodes(newNodes);
      setEdges(newEdges);
    }
  }, [layerData]);

  const loadModels = async () => {
    try {
      console.log("[Layer Graph View] Fetching models from /api/models");
      const response = await fetch("/api/models");
      console.log(
        "[Layer Graph View] Models response status:",
        response.status
      );

      if (!response.ok) {
        console.error(
          "[Layer Graph View] Models fetch failed with status:",
          response.status
        );
        setLoading(false);
        return;
      }

      const data = await response.json();
      console.log("[Layer Graph View] Received models:", data.models);
      setModels(data.models || []);

      if (data.models && data.models.length > 0) {
        console.log(
          "[Layer Graph View] Auto-selecting first model:",
          data.models[0]
        );
        setSelectedModel(data.models[0]);
      } else {
        console.warn("[Layer Graph View] No models found in response");
      }
    } catch (error) {
      console.error("[Layer Graph View] Error loading models:", error);
    } finally {
      setLoading(false);
    }
  };

  const loadLayerData = async (model: string) => {
    setLoadingGraph(true);
    try {
      console.log(`[Layer Graph View] Fetching layer data for model: ${model}`);
      const response = await fetch(`/api/layers/${model}`);
      console.log(
        "[Layer Graph View] Layer data response status:",
        response.status
      );

      if (!response.ok) {
        console.error(
          "[Layer Graph View] Layer data fetch failed with status:",
          response.status
        );
        return;
      }

      const data = await response.json();
      console.log(`[Layer Graph View] Received layer data for ${model}:`, {
        layersCount: data.layers?.length || 0,
        visualizationsCount: data.visualizations?.length || 0,
      });

      setLayerData(data);
    } catch (error) {
      console.error("[Layer Graph View] Error loading layer data:", error);
    } finally {
      setLoadingGraph(false);
    }
  };

  const onNodesChange = useCallback(
    (changes: NodeChange[]) =>
      setNodes((nds) => applyNodeChanges(changes, nds)),
    []
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) =>
      setEdges((eds) => applyEdgeChanges(changes, eds)),
    []
  );

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <Spinner size="lg" />
        <p className="text-muted-foreground">Loading models...</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Model Selector */}
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

      {/* Graph View */}
      <div className="col-span-10">
        <div className="border rounded-md">
          <div className="p-4 border-b flex items-center justify-between">
            <div>
              <h3 className="font-semibold">Layer Flow Graph</h3>
              {layerData && (
                <p className="text-xs text-muted-foreground mt-1">
                  {selectedModel} - {layerData.layers.length} layers
                </p>
              )}
            </div>
            <div className="text-xs text-muted-foreground">
              Scroll to zoom • Drag to pan • Click nodes for details
            </div>
          </div>
          <div style={{ height: "600px" }}>
            {loadingGraph ? (
              <div className="flex flex-col items-center justify-center h-full space-y-4">
                <Spinner size="lg" />
                <p className="text-muted-foreground">Loading layer graph...</p>
              </div>
            ) : nodes.length > 0 ? (
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                nodeTypes={nodeTypes}
                connectionMode={ConnectionMode.Strict}
                fitView
                fitViewOptions={{ padding: 0.2 }}
                minZoom={0.1}
                maxZoom={1.5}
                defaultViewport={{ x: 0, y: 0, zoom: 0.5 }}
                className="bg-background"
              >
                <Background color="#333" gap={16} />
                <Controls className="!bg-card [&_button]:!bg-card [&_button]:!border-border [&_button]:!text-foreground [&_button:hover]:!bg-secondary" />
                <MiniMap
                  nodeStrokeWidth={3}
                  zoomable
                  pannable
                  className="!bg-card !border-border"
                  maskColor="rgba(0, 0, 0, 0.6)"
                />
              </ReactFlow>
            ) : (
              <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">
                  Select a model to view layer graph
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
