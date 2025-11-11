"use client";

import { memo } from "react";
import { Handle, Position } from "@xyflow/react";

export type LayerNodeData = {
  idx: number;
  name: string;
  type: string;
  shape: number[];
  min: number | null;
  max: number | null;
  mean: number | null;
  std: number | null;
  sparsity: number | null;
  visualizationPath: string;
};

interface LayerNodeProps {
  data: LayerNodeData;
  selected?: boolean;
}

function LayerNode({ data, selected }: LayerNodeProps) {
  const layer = data;

  return (
    <div
      className={`border-2 rounded-lg bg-card overflow-hidden transition-all ${
        selected ? "border-primary shadow-lg" : "border-border"
      }`}
      style={{ minWidth: "250px" }}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="w-3 h-3 !bg-primary"
        isConnectable={true}
      />

      {/* Layer Image */}
      {layer.visualizationPath && (
        <div className="relative w-full h-32 bg-muted">
          <img
            src={
              layer.visualizationPath.startsWith("https://")
                ? layer.visualizationPath
                : `/api/viz${layer.visualizationPath}`
            }
            alt={`Layer ${layer.idx} visualization`}
            className="w-full h-full object-cover"
            loading="lazy"
            decoding="async"
          />
        </div>
      )}

      {/* Layer Info */}
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span className="text-xs font-semibold text-muted-foreground">
              Layer {layer.idx}
            </span>
          </div>
          <span className="text-xs px-2 py-0.5 rounded bg-secondary font-mono">
            {layer.type}
          </span>
        </div>

        <div>
          <div className="font-semibold text-sm truncate" title={layer.name}>
            {layer.name}
          </div>
          <div className="text-xs text-muted-foreground font-mono">
            [{layer.shape.join(" × ")}]
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-1 text-xs pt-2 border-t">
          {layer.mean !== null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Mean:</span>
              <span className="font-mono">{layer.mean.toFixed(3)}</span>
            </div>
          )}
          {layer.std !== null && (
            <div className="flex justify-between">
              <span className="text-muted-foreground">Std:</span>
              <span className="font-mono">{layer.std.toFixed(3)}</span>
            </div>
          )}
          {layer.sparsity !== null && (
            <div className="flex justify-between col-span-2">
              <span className="text-muted-foreground">Sparsity:</span>
              <span className="font-mono">
                {(layer.sparsity * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="w-3 h-3 !bg-primary"
        isConnectable={true}
      />
    </div>
  );
}

export default memo(LayerNode);
