"use client";

import { useState, useEffect } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BenchmarkExplorer } from "@/components/benchmark-explorer";
import { LayerExplorer } from "@/components/layer-explorer";
import { MetricsMonitor } from "@/components/metrics-monitor";

export default function Home() {
  const [activeTab, setActiveTab] = useState("benchmarks");

  return (
    <main className="min-h-screen bg-background">
      <div className="border-b">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold tracking-tight">
            Vision-Dissect Explorer
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            Research tool for model benchmarks and layer visualization
          </p>
        </div>
      </div>

      <div className="container mx-auto px-4 py-6">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-secondary">
            <TabsTrigger value="benchmarks">Benchmarks</TabsTrigger>
            <TabsTrigger value="layers">Layer Visualizations</TabsTrigger>
            <TabsTrigger value="monitor">Live Monitor</TabsTrigger>
          </TabsList>

          <TabsContent value="benchmarks" className="mt-6">
            <BenchmarkExplorer />
          </TabsContent>

          <TabsContent value="layers" className="mt-6">
            <LayerExplorer />
          </TabsContent>

          <TabsContent value="monitor" className="mt-6">
            <MetricsMonitor />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  );
}
