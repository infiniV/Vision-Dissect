"use client";

import { useState, useEffect } from "react";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";

interface BenchmarkRun {
  timestamp: string;
  models: string[];
  path: string;
}

interface ModelMetrics {
  model: string;
  loadTime: number;
  avgInference: number;
  std: number;
  fps: number;
  peakMemory: number;
  parameters: number;
  layers: number;
}

export function BenchmarkExplorer() {
  const [benchmarkRuns, setBenchmarkRuns] = useState<BenchmarkRun[]>([]);
  const [selectedRun, setSelectedRun] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  useEffect(() => {
    loadBenchmarkRuns();
  }, []);

  useEffect(() => {
    if (selectedRun) {
      loadMetrics(selectedRun);
    }
  }, [selectedRun]);

  const loadBenchmarkRuns = async () => {
    try {
      console.log(
        "[Benchmark Explorer] Fetching benchmark runs from /api/benchmarks"
      );
      const response = await fetch("/api/benchmarks");
      console.log(
        "[Benchmark Explorer] Runs response status:",
        response.status
      );

      if (!response.ok) {
        console.error(
          "[Benchmark Explorer] Runs fetch failed with status:",
          response.status
        );
        setLoading(false);
        return;
      }

      const data = await response.json();
      console.log(
        "[Benchmark Explorer] Received runs:",
        data.runs?.map((r: BenchmarkRun) => ({
          timestamp: r.timestamp,
          modelCount: r.models.length,
        }))
      );

      setBenchmarkRuns(data.runs || []);
      if (data.runs && data.runs.length > 0) {
        console.log(
          "[Benchmark Explorer] Auto-selecting first run:",
          data.runs[0].timestamp
        );
        setSelectedRun(data.runs[0].timestamp);
      } else {
        console.warn("[Benchmark Explorer] No runs found in response");
      }
    } catch (error) {
      console.error(
        "[Benchmark Explorer] Error loading benchmark runs:",
        error
      );
    } finally {
      setLoading(false);
    }
  };

  const loadMetrics = async (timestamp: string) => {
    setLoadingMetrics(true);
    try {
      console.log(
        `[Benchmark Explorer] Fetching metrics for timestamp: ${timestamp}`
      );
      const response = await fetch(`/api/benchmarks/${timestamp}`);
      console.log(
        "[Benchmark Explorer] Metrics response status:",
        response.status
      );

      if (!response.ok) {
        console.error(
          "[Benchmark Explorer] Metrics fetch failed with status:",
          response.status
        );
        return;
      }

      const data = await response.json();
      console.log(
        `[Benchmark Explorer] Received metrics count: ${
          data.metrics?.length || 0
        }`
      );

      if (data.metrics && data.metrics.length > 0) {
        console.log("[Benchmark Explorer] First metric:", data.metrics[0]);
        console.log(
          "[Benchmark Explorer] All model names:",
          data.metrics.map((m: ModelMetrics) => m.model)
        );
      } else {
        console.warn("[Benchmark Explorer] No metrics found in response");
      }

      setMetrics(data.metrics || []);
    } catch (error) {
      console.error("[Benchmark Explorer] Error loading metrics:", error);
    } finally {
      setLoadingMetrics(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <Spinner size="lg" />
        <p className="text-muted-foreground">Loading benchmarks...</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-3 border rounded-md">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Benchmark Runs</h3>
        </div>
        <ScrollArea className="h-[600px]">
          <div className="p-2">
            {benchmarkRuns.map((run) => (
              <button
                key={run.timestamp}
                onClick={() => setSelectedRun(run.timestamp)}
                className={`w-full text-left p-3 rounded border mb-2 transition-colors ${
                  selectedRun === run.timestamp
                    ? "bg-secondary border-primary"
                    : "border-border hover:border-primary"
                }`}
              >
                <div className="text-sm font-medium">{run.timestamp}</div>
                <div className="text-xs text-muted-foreground mt-1">
                  {run.models.length} models
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      <div className="col-span-9">
        <div className="border rounded-md">
          <div className="p-4 border-b">
            <h3 className="font-semibold">Performance Metrics</h3>
            {selectedRun && (
              <p className="text-xs text-muted-foreground mt-1">
                Run: {selectedRun}
              </p>
            )}
          </div>
          <ScrollArea className="h-[600px]">
            <div className="p-4">
              {loadingMetrics ? (
                <div className="flex flex-col items-center justify-center py-16 space-y-4">
                  <Spinner size="lg" />
                  <p className="text-sm text-muted-foreground">
                    Loading metrics...
                  </p>
                </div>
              ) : metrics.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-8">
                  No metrics available
                </p>
              ) : (
                <Accordion type="single" collapsible className="w-full">
                  {metrics.map((model, idx) => (
                    <AccordionItem key={idx} value={`item-${idx}`}>
                      <AccordionTrigger className="hover:no-underline">
                        <div className="flex items-center justify-between w-full pr-4">
                          <span className="font-medium">{model.model}</span>
                          <span className="text-sm text-muted-foreground">
                            {model.fps.toFixed(2)} FPS
                          </span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Load Time
                              </span>
                              <span className="font-mono">
                                {model.loadTime.toFixed(3)}s
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Avg Inference
                              </span>
                              <span className="font-mono">
                                {model.avgInference.toFixed(3)}s
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Std Dev
                              </span>
                              <span className="font-mono">
                                {model.std.toFixed(3)}s
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">FPS</span>
                              <span className="font-mono">
                                {model.fps.toFixed(2)}
                              </span>
                            </div>
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Peak Memory
                              </span>
                              <span className="font-mono">
                                {model.peakMemory} MB
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Parameters
                              </span>
                              <span className="font-mono">
                                {(model.parameters / 1e6).toFixed(2)}M
                              </span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">
                                Layers
                              </span>
                              <span className="font-mono">{model.layers}</span>
                            </div>
                          </div>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              )}
            </div>
          </ScrollArea>
        </div>
      </div>
    </div>
  );
}
