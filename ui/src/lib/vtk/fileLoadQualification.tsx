import { createRoot } from "react-dom/client";
import { PlotStudio } from "@/components/studio/PlotStudio";
import { invokeCommand } from "@/lib/platform/appInvoke";
import { getHardwareInfo, getProcessMemoryInfo } from "@/lib/platform/tauri";
import { listSteps } from "@/lib/run/runOutputs";
import { fieldCacheStats } from "@/lib/core/fieldCache";
import { fieldLoadSchedulerStats } from "@/lib/core/loadScheduler";
import { getVisualizationResourceSnapshot } from "@/lib/core/visualizationResourceGovernor";
import type { PanePresentationSnapshot } from "@/lib/plot/panePresentationBarrier";
import { preparedDatasetRegistrySnapshot } from "./vtkPreparedDatasetRegistry";
import { lookupVtkAppearanceCapture, lookupVtkCapture, lookupVtkDescriptor, lookupVtkRenderWindow } from "./vtkCaptureRegistry";
import { holdVtkFullQuality } from "./vtkInteraction";
import type { PackagedVisualizationQualificationRequest } from "./packagedVisualizationQualification";

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms));
const progress = (stage: string) => invokeCommand("record_visualization_qualification_progress", { stage });
const canvases = () => {
  const seen = new Set();
  return [...document.querySelectorAll<HTMLCanvasElement>(".volume-render__canvas canvas")].filter(canvas => {
    const window = lookupVtkRenderWindow(canvas);
    if (!window || seen.has(window)) return false;
    seen.add(window);
    return true;
  });
};
const descriptors = () => canvases().map(canvas => lookupVtkDescriptor(canvas)!());

async function until(description: string, predicate: () => boolean, timeout = 180_000): Promise<void> {
  const start = performance.now();
  while (!predicate()) {
    const failed = document.querySelector(".error-boundary__message");
    if (failed) throw new Error(`${description}: ${failed.textContent}`);
    const fieldFailure = document.querySelector(".volume-render__overlay--error");
    if (fieldFailure) throw new Error(`${description}: ${fieldFailure.textContent}`);
    if (performance.now() - start > timeout) {
      throw new Error(`${description} timed out: ${JSON.stringify({ descriptors: descriptors(),
        scheduler: fieldLoadSchedulerStats(), text: document.body.innerText.slice(-8000) })}`);
    }
    await sleep(100);
  }
}

function button(group: string, text: string): HTMLButtonElement {
  const result = [...document.querySelectorAll<HTMLButtonElement>(`${group} button`)]
    .find(item => item.textContent?.trim() === text);
  if (!result || result.disabled) throw new Error(`Control unavailable: ${text}`);
  return result;
}

function setValue(element: HTMLInputElement | HTMLSelectElement, value: string): void {
  const prototype = element instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(element, value);
  element.dispatchEvent(new Event("input", { bubbles: true }));
  element.dispatchEvent(new Event("change", { bubbles: true }));
}

function gpuIdentity(canvas: HTMLCanvasElement) {
  const gl = canvas.getContext("webgl2") ?? canvas.getContext("webgl");
  if (!gl) return null;
  const extension = gl.getExtension("WEBGL_debug_renderer_info");
  return { vendor: gl.getParameter(extension?.UNMASKED_VENDOR_WEBGL ?? gl.VENDOR),
    renderer: gl.getParameter(extension?.UNMASKED_RENDERER_WEBGL ?? gl.RENDERER),
    version: gl.getParameter(gl.VERSION), max3DTextureSize: "MAX_3D_TEXTURE_SIZE" in gl
      ? gl.getParameter((gl as WebGL2RenderingContext).MAX_3D_TEXTURE_SIZE) : null };
}

function percentile(values: number[], fraction: number) {
  return [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(values.length * fraction))] ?? null;
}

function call(object: unknown, method: string): unknown {
  if (!object || typeof object !== "object") return null;
  const fn = (object as Record<string, unknown>)[method];
  return typeof fn === "function" ? fn.call(object) : null;
}

function dataGeometry(canvas: HTMLCanvasElement) {
  const renderer = lookupVtkRenderWindow(canvas)?.getRenderer();
  return ["getVolumes", "getActors"].flatMap(kind => {
    const layers = call(renderer, kind);
    return Array.isArray(layers) ? layers.map(layer => {
      const data = call(call(layer, "getMapper"), "getInputData");
      const scalars = call(call(data, "getPointData"), "getScalars");
      const values = call(scalars, "getData");
      return { kind, visible: call(layer, "getVisibility"), dimensions: call(data, "getDimensions"),
        spacing: call(data, "getSpacing"), origin: call(data, "getOrigin"), bounds: call(data, "getBounds"),
        components: call(scalars, "getNumberOfComponents"), scalarBytes: ArrayBuffer.isView(values) ? values.byteLength : null };
    }) : [];
  });
}

export async function runFileLoadQualification(request: PackagedVisualizationQualificationRequest): Promise<void> {
  const started = performance.now();
  const checks: { name: string; passed: boolean; detail: string }[] = [];
  const scenes: Record<string, unknown>[] = [];
  const transitions: Record<string, unknown>[] = [];
  const watchdog: { elapsedMs: number; delayMs: number; stage: string }[] = [];
  const storage = Object.fromEntries(Object.keys(localStorage).map(key => [key, localStorage.getItem(key)!]));
  const root = createRoot(document.getElementById("root")!);
  const inputEvents = ["pointerdown", "pointermove", "pointerup", "mousedown", "mousemove", "mouseup",
    "touchstart", "touchmove", "touchend", "keydown", "click", "input", "change", "wheel"];
  const holdUserInput = (event: Event) => {
    if (event.isTrusted) { event.preventDefault(); event.stopImmediatePropagation(); }
  };
  for (const name of inputEvents) window.addEventListener(name, holdUserInput, { capture: true, passive: false });
  let barrier: PanePresentationSnapshot | null = null;
  let frameIndices: number[] = [];
  let stage = "discovery";
  const glCalls: { method: string; elapsedMs: number; stage: string }[] = [];
  const restoreGl: (() => void)[] = [];
  for (const prototype of [WebGLRenderingContext.prototype, WebGL2RenderingContext.prototype]) {
    for (const method of ["texImage3D", "texSubImage3D", "texImage2D", "texSubImage2D", "compileShader", "linkProgram", "getProgramParameter", "readPixels", "drawArrays", "drawElements", "bufferData", "finish"]) {
      const descriptor = Object.getOwnPropertyDescriptor(prototype, method);
      if (!descriptor || typeof descriptor.value !== "function") continue;
      const original = descriptor.value;
      Object.defineProperty(prototype, method, { ...descriptor, value: function (this: unknown, ...args: unknown[]) {
        const start = performance.now();
        try { return Reflect.apply(original, this, args); }
        finally {
          const elapsedMs = performance.now() - start;
          if (elapsedMs >= 8) glCalls.push({ method, elapsedMs, stage });
        }
      } });
      restoreGl.push(() => Object.defineProperty(prototype, method, descriptor));
    }
  }
  let lastTick = performance.now();
  const timer = setInterval(() => {
    const now = performance.now();
    if (now - lastTick > 125) watchdog.push({ elapsedMs: now - started, delayMs: now - lastTick - 50, stage });
    lastTick = now;
  }, 50);
  const report: Record<string, unknown> = {
    schema: "nimbus.studio.packaged-visualization-qualification", schemaVersion: 3,
    scope: "bounded-packaged-lifecycle", memoryCampaign: "pending", renderPerformanceCampaign: "pending",
    profile: "file-load", fixture: request.visualLoad, checks, scenes, transitions, watchdog, glCalls,
    userAgent: navigator.userAgent, status: "failed",
  };
  const check = (name: string, passed: boolean, detail: string) => {
    checks.push({ name, passed, detail });
    if (!passed) throw new Error(`${name}: ${detail}`);
  };
  const idle = (plot: string, panes: number, frame: number) => {
    const current = descriptors();
    const recorder = document.querySelector<HTMLButtonElement>(".plot-studio__rec-btn");
    return current.length === panes && current.every(d => d.plot === plot && d.frameIndex === frameIndices[frame])
      && recorder != null && !recorder.disabled
      && fieldLoadSchedulerStats().active === 0 && fieldLoadSchedulerStats().queued === 0
      && !document.querySelector('[data-scene-frame-commit-pending="true"], [data-scene-presentation-pending="true"]')
      && (panes === 1 || (barrier?.allowed === true && barrier.readyPaneIds.length === panes));
  };
  async function frame(plot: string, panes: number, index: number) {
    stage = `${panes}-${plot}-frame-${index}`;
    await progress(stage);
    const start = performance.now();
    const input = document.querySelector<HTMLInputElement>('input[aria-label="Selected frame"]');
    if (!input) throw new Error("Frame slider missing");
    const previous = descriptors();
    const legendCount = previous.length === panes && previous.every(d => d.plot === plot)
      ? document.querySelectorAll(".plot3d-colorbar").length : 0;
    let missingLegendSamples = 0;
    const legendTimer = setInterval(() => {
      if (document.querySelectorAll(".plot3d-colorbar").length < legendCount) missingLegendSamples++;
    }, 20);
    try {
      setValue(input, String(index));
      await sleep(100);
      await until(stage, () => idle(plot, panes, index));
    } finally { clearInterval(legendTimer); }
    if (legendCount > 0) check(`${stage}-legends`, missingLegendSamples === 0,
      `${legendCount} legends retained during frame loading; ${missingLegendSamples} missing samples`);
    transitions.push({ stage, frame: index, panes, elapsedMs: performance.now() - start,
      barrier, descriptors: descriptors(), cache: fieldCacheStats(), scheduler: fieldLoadSchedulerStats() });
  }
  async function capture(name: string, repeat = false) {
    stage = name;
    await progress(stage);
    const captures = [];
    for (const canvas of canvases()) {
      const releaseQuality = holdVtkFullQuality(lookupVtkRenderWindow(canvas)!);
      await sleep(250);
      const take = lookupVtkAppearanceCapture(canvas)!;
      let first;
      try {
        first = await take();
        if (repeat) {
          const save = lookupVtkCapture(canvas)!;
          const a = await save();
          const b = await save();
          report.exportPair = { first: a, second: b, descriptor: lookupVtkDescriptor(canvas)!() };
          check("steady-export", a === b, "Two ordinary full-quality exports of the same committed frame are identical");
        }
      } finally { releaseQuality(); }
      check(`${name}-visible-${captures.length}`, first.evidence.layers.some(layer => layer.visible && (layer.pointCount ?? 0) > 0),
        "Committed scene contains visible data");
      captures.push({ ...first, gpu: gpuIdentity(canvas), geometry: dataGeometry(canvas),
        sceneBudget: { ...canvas.closest<HTMLElement>(".volume-render__canvas-wrap")?.dataset } });
    }
    async function orbit(interactive: boolean) {
      const intervals: number[] = [];
      const windows = canvases().map(canvas => lookupVtkRenderWindow(canvas)!);
      const styles = windows.map(window => call(call(window, "getInteractor"), "getInteractorStyle"));
      if (interactive) for (const style of styles) call(style, "invokeStartInteractionEvent");
      let previous = performance.now();
      try {
        for (let i = 0; i < 24; i++) {
          await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
          for (const window of windows) {
            const renderer = window.getRenderer() as { getActiveCamera(): { azimuth(value: number): void } };
            renderer.getActiveCamera().azimuth(1);
            window.getRenderWindow().render();
          }
          const now = performance.now(); intervals.push(now - previous); previous = now;
        }
      } finally {
        if (interactive) for (const style of styles) call(style, "invokeEndInteractionEvent");
      }
      return { intervalsMs: intervals, medianMs: percentile(intervals, .5), p95Ms: percentile(intervals, .95),
        mode: interactive ? "ordinary interaction preview" : "programmatic orbit without gesture events", renderedFrames: intervals.length };
    }
    const settledOrbit = await orbit(false);
    const interactiveOrbit = await orbit(true);
    scenes.push({ name, captures, orbit: settledOrbit, interactiveOrbit,
      memory: await getProcessMemoryInfo(), resources: getVisualizationResourceSnapshot(),
      prepared: preparedDatasetRegistrySnapshot(), cache: fieldCacheStats() });
  }
  try {
    if (!request.visualLoad?.runDir || request.visualLoad.error) throw new Error(request.visualLoad?.error ?? "Missing fixture directory");
    report.hardware = await getHardwareInfo();
    const steps = await listSteps(request.visualLoad.runDir);
    frameIndices = steps.map(step => step.index);
    report.steps = steps.map(step => ({ index: step.index, fields: step.fields, grid: step.grid }));
    check("native-history", steps.length === request.visualLoad.specification?.frames
      && steps.every(step => ["cloud_condensate", "qr", "w"].every(field => step.fields.includes(field)) && step.grid != null),
      "Ordinary native history reader reopens every frame with cloud, precipitation, velocity, and physical geometry");
    root.render(<div style={{ height: "100vh", display: "flex", flexDirection: "column" }}>
      <style>{".plot-studio { flex: 1; min-height: 0; }"}</style>
      <div style={{ padding: "6px 16px", background: "#594516", color: "#fff" }}>
        AUTOMATED RENDERER TEST · {request.visualLoad.specification?.synthetic === false
          ? "SAVED STORM REFERENCE — ORIGINAL RESOLUTION" : "SYNTHETIC DATA — NOT A SIMULATION"} · {JSON.stringify(request.visualLoad.specification?.shape)}
      </div>
      <PlotStudio restoreActiveProject={false} initialRunDir={request.visualLoad.runDir}
        onPanePresentationChange={snapshot => { barrier = snapshot; }} />
    </div>);
    await until("dataset controls", () => !!document.querySelector('[aria-label="Plot categories"] button'));
    button('[aria-label="Plot categories"]', "3D").click();
    await sleep(100);
    for (const panes of [1, 4]) {
      if (panes === 4) {
        document.querySelector<HTMLElement>(".plot-studio__view-menu summary")!.click();
        await sleep(100);
        button('[aria-label="Pane layout"]', "Quad").click();
        await until("companion controls", () => document.querySelectorAll('[aria-label="Companion pane plot type"]').length === 3);
      }
      for (const plot of ["volume", "supercell"]) {
        stage = `${panes === 1 ? "single" : "quad"}-${plot}`;
        await progress(stage);
        const start = performance.now();
        button('[aria-label="Plot types"]', plot === "volume" ? "Volume" : "Rendered supercell").click();
        for (const select of document.querySelectorAll<HTMLSelectElement>('[aria-label="Companion pane plot type"]')) setValue(select, plot);
        await frame(plot, panes, 0);
        for (const tab of document.querySelectorAll<HTMLButtonElement>('button[aria-controls$="-scene"]')) {
          if (tab.getAttribute("aria-expanded") !== "true") tab.click();
        }
        await until("scene controls", () => document.querySelectorAll('[aria-label="Horizontal scene scale"]').length === panes, 10_000);
        for (const label of ["Horizontal scene scale", "Vertical scene scale"]) {
          for (const input of document.querySelectorAll<HTMLInputElement>(`[aria-label="${label}"]`)) setValue(input, "1");
          await sleep(100);
        }
        for (const control of document.querySelectorAll<HTMLButtonElement>(".plot3d-control-panel button")) {
          if (control.textContent?.trim() === "Reset lighting" || control.textContent?.trim() === "Reset camera") control.click();
        }
        await sleep(500);
        await until("scene settled", () => idle(plot, panes, 0));
        if (plot === "volume") {
          for (const tab of document.querySelectorAll<HTMLButtonElement>('[aria-label="Volume controls"] button[aria-controls$="-volume"]')) {
            if (tab.getAttribute("aria-expanded") !== "true") tab.click();
          }
          await until("field controls", () => document.querySelectorAll('select[aria-label="Field"]').length === panes, 10_000);
          for (const select of document.querySelectorAll<HTMLSelectElement>('select[aria-label="Field"]')) setValue(select, "cloud_condensate");
          await sleep(100);
          await until("cloud field", () => idle(plot, panes, 0) && descriptors().every(d => d.field === "cloud_condensate"));
        } else {
          if (panes === 1) await capture("single-supercell-cloud");
          for (const group of ["precip", "flow"]) {
            stage = `${panes}-supercell-enable-${group}`;
            await progress(stage);
            for (const tab of document.querySelectorAll<HTMLButtonElement>(`button[aria-controls="supercell-controls-${group}"]`)) tab.click();
            await sleep(100);
            const toggles = [...document.querySelectorAll(`#supercell-controls-${group}`)]
              .map(panel => panel.querySelector<HTMLButtonElement>("button.pill"));
            if (toggles.length !== panes || toggles.some(toggle => !toggle)) throw new Error(`Missing ${group} controls in ${panes}-pane scene`);
            for (const toggle of toggles) if (toggle?.textContent?.trim() === "off") toggle.click();
            await sleep(100);
            await until(`${group} layers`, () => idle(plot, panes, 0));
          }
          check(`${panes}-supercell-multifield`, descriptors().every(d => d.parameters?.showPrecip === true && d.parameters?.showUpdraft === true),
            "Cloud, precipitation and updraft are enabled in every supercell pane");
        }
        const name = `${panes === 1 ? "single" : "quad"}-${plot}`;
        check(name, true, `Initial committed scene after ${(performance.now() - start).toFixed(1)} ms`);
        await capture(name, panes === 1 && plot === "supercell");
        for (const index of [1, 2, 3, 0]) await frame(plot, panes, index % steps.length);
      }
    }
    stage = "quad-supercell-playback";
    await progress(stage);
    const seen = new Set<number>();
    const incoherent: unknown[] = [];
    document.querySelector<HTMLButtonElement>('[aria-label="Play frames"]')!.click();
    const deadline = performance.now() + 120_000;
    while (seen.size < Math.min(3, steps.length) && performance.now() < deadline) {
      await sleep(100);
      const shown = descriptors();
      if (shown.length === 4) {
        const frames = shown.map(d => d.frameIndex);
        if (frames.every(value => value === frames[0]) && typeof frames[0] === "number") seen.add(frames[0]);
        else incoherent.push(frames);
      }
    }
    document.querySelector<HTMLButtonElement>('[aria-label="Pause frame playback"]')?.click();
    report.playback = { framesSeen: [...seen], incoherent, requestedFps: document.querySelector<HTMLSelectElement>('[aria-label="Playback speed"]')?.value };
    check("playback-sync", seen.size >= Math.min(3, steps.length) && incoherent.length === 0,
      "All four committed scenes advance through at least three frames without mixed frame indices");
    report.status = "passed";
  } catch (error) {
    report.error = error instanceof Error ? { message: error.message, stack: error.stack } : String(error);
  } finally {
    clearInterval(timer);
    report.elapsedMs = performance.now() - started;
    root.unmount();
    for (const restore of restoreGl) restore();
    for (const name of inputEvents) window.removeEventListener(name, holdUserInput, { capture: true });
    for (const key of Object.keys(localStorage)) if (!(key in storage)) localStorage.removeItem(key);
    for (const [key, value] of Object.entries(storage)) localStorage.setItem(key, value);
  }
  await invokeCommand("finish_visualization_qualification", { report });
}
