"use client";

import { useState, useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";

interface ProcessStatus {
  running: boolean;
  currentModel: string | null;
  progress: number;
  totalModels: number;
  startTime: string | null;
  logs: string[];
}

export function MetricsMonitor() {
  const [status, setStatus] = useState<ProcessStatus>({
    running: false,
    currentModel: null,
    progress: 0,
    totalModels: 0,
    startTime: null,
    logs: [],
  });

  useEffect(() => {
    const interval = setInterval(checkStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const checkStatus = async () => {
    try {
      console.log("[Metrics Monitor] Fetching status from /api/status");
      const response = await fetch("/api/status");
      console.log("[Metrics Monitor] Status response status:", response.status);

      if (!response.ok) {
        console.error(
          "[Metrics Monitor] Status fetch failed with status:",
          response.status
        );
        return;
      }

      const data = await response.json();
      console.log("[Metrics Monitor] Received status data:", {
        running: data.running,
        currentModel: data.currentModel,
        progress: data.progress,
        totalModels: data.totalModels,
        logsCount: data.logs?.length || 0,
      });
      setStatus(data);
    } catch (error) {
      console.error("[Metrics Monitor] Error checking status:", error);
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        <div className="border rounded-md p-4">
          <div className="text-sm text-muted-foreground mb-1">Status</div>
          <div
            className={`text-lg font-semibold ${
              status.running ? "text-green-600" : ""
            }`}
          >
            {status.running ? "Running" : "Idle"}
          </div>
        </div>

        <div className="border rounded-md p-4">
          <div className="text-sm text-muted-foreground mb-1">
            Current Model
          </div>
          <div className="text-lg font-semibold truncate">
            {status.currentModel || "None"}
          </div>
        </div>

        <div className="border rounded-md p-4">
          <div className="text-sm text-muted-foreground mb-1">Progress</div>
          <div className="text-lg font-semibold">
            {status.progress} / {status.totalModels}
          </div>
        </div>

        <div className="border rounded-md p-4">
          <div className="text-sm text-muted-foreground mb-1">Start Time</div>
          <div className="text-sm font-mono">{status.startTime || "N/A"}</div>
        </div>
      </div>

      <div className="border rounded-md">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Process Logs</h3>
        </div>
        <ScrollArea className="h-[500px]">
          <div className="p-4 font-mono text-xs space-y-1">
            {status.logs.length === 0 ? (
              <p className="text-muted-foreground">No logs available</p>
            ) : (
              status.logs.map((log, idx) => (
                <div key={idx} className="py-1">
                  {log}
                </div>
              ))
            )}
          </div>
        </ScrollArea>
      </div>
    </div>
  );
}
