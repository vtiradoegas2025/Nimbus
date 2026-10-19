import { createRoot } from "react-dom/client";
import { App } from "./App";
import { installStudioAppearance } from "@/lib/studio/studioAppearance";
import { getHardwareInfo, isTauriRuntime } from "@/lib/platform/tauri";
import { visualizationResourceGovernor } from "@/lib/core/visualizationResourceGovernor";
import { initializeStudioPerformance } from "@/lib/studio/studioPerformance";
import {
  installVisualizationCapacityQualification,
  parseVisualizationCapacityQualification,
  resolveVisualizationCapacity,
} from "@/lib/core/visualizationCapacityPolicy";
import { invokeCommand } from "@/lib/platform/appInvoke";
import type { PackagedVisualizationQualificationRequest } from "@/lib/vtk/packagedVisualizationQualification";
import "./styles/theme.css";
import "./styles/base.css";

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("Root element #root not found in index.html");
}

// Resolve appearance before React mounts so the first painted frame uses the
// persisted light/dark/system choice. The controller remains for the app
// lifetime to follow live operating-system changes.
installStudioAppearance();

async function start(root: HTMLElement): Promise<void> {
  if (isTauriRuntime()) {
    await getHardwareInfo().then(async (hardware) => {
      const facts = { source: "tauri", systemRamBytes: hardware.ramBytes,
        logicalCores: hardware.coreCount } as const;
      visualizationResourceGovernor.setHardwareFacts(facts);
      await initializeStudioPerformance(hardware, true);
      const capacity = await invokeCommand<unknown>("visualization_capacity_qualification")
        .then(parseVisualizationCapacityQualification);
      installVisualizationCapacityQualification(capacity);
      const decision = resolveVisualizationCapacity(facts, capacity);
      if (decision.mode === "enforced") {
        visualizationResourceGovernor.setCapacity({ totalBytes: decision.capacityBytes });
      }
    }).catch(() => undefined);
    const qualification = await invokeCommand<PackagedVisualizationQualificationRequest | null>(
      "visualization_qualification_request",
    ).catch(() => null);
    if (qualification) {
      await invokeCommand("record_visualization_qualification_progress", {
        stage: "frontend-startup",
      }).catch(() => undefined);
      if (qualification.profile === "file-load") {
        const { runFileLoadQualification } = await import("@/lib/vtk/fileLoadQualification");
        await runFileLoadQualification(qualification);
        return;
      }
      const { runPackagedVisualizationQualification } = await import(
        "@/lib/vtk/packagedVisualizationQualification"
      );
      await invokeCommand("record_visualization_qualification_progress", {
        stage: "qualification-module-loaded",
      }).catch(() => undefined);
      await runPackagedVisualizationQualification(qualification);
      return;
    }
  } else {
    const cores = navigator.hardwareConcurrency;
    const facts = {
      source: "browser",
      systemRamBytes: null,
      logicalCores: Number.isSafeInteger(cores) && cores > 0 ? cores : null,
    } as const;
    visualizationResourceGovernor.setHardwareFacts(facts);
    await initializeStudioPerformance({ cpuBrand: null, coreCount: facts.logicalCores, ramBytes: null }, false);
  }

  // StrictMode omitted: double-mount in dev tears down vtk.js/WebGL contexts, causing 3D flicker and context loss.
  createRoot(root).render(<App />);
}

void start(rootEl).catch((error: unknown) => {
  rootEl.textContent = error instanceof Error ? error.message : String(error);
  throw error;
});
