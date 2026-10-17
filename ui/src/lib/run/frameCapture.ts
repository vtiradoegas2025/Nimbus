/*
 * Capture the active plot panel (Plotly or VTK) as a PNG data URI.
 * 3D widgets use vtkCaptureRegistry because preserveDrawingBuffer=false makes toDataURL() blank.
 */

import Plotly from "plotly.js-dist-min";
import { lookupVtkCapture, lookupVtkRenderWindow } from "@/lib/vtk/vtkCaptureRegistry";
import { holdVtkFullQuality } from "@/lib/vtk/vtkInteraction";
import { holdFullVtkPlaybackResolution } from "@/lib/vtk/vtkPlaybackScreenResolution";
import { assertCaptureCurrent, guardedCapture, type CaptureGuard } from "./captureGuard";
import { isPlotCanvasReceiptCurrent, waitForPlotCanvasReady, type PlotCanvasReceipt } from "@/lib/plot/plotCanvasReadiness";

export interface CaptureSize {
  width: number;
  height: number;
}

/** Capture the plot panel at `el` as a PNG `data:image/png;base64,…`
 *  data URI. Throws on capture failure; the caller surfaces the message
 *  in the recording dialog. */
export async function captureFrameDataUri(
  el: HTMLElement,
  size: CaptureSize,
  guard: CaptureGuard = {},
): Promise<string> {
  if (!Number.isSafeInteger(size.width) || !Number.isSafeInteger(size.height) || size.width < 1 || size.height < 1) throw new Error("Capture dimensions must be positive integers");
  const vtkCapture = lookupVtkCapture(el);
  let receipt: PlotCanvasReceipt | null = null;
  const viewport = () => {
    const box = el.getBoundingClientRect();
    return JSON.stringify([box.width, box.height, el.ownerDocument.defaultView?.devicePixelRatio ?? 1]);
  };
  const backing = () => {
    const canvas = el.tagName === "CANVAS" ? el as HTMLCanvasElement : el.querySelector("canvas");
    return { canvas, width: canvas?.width, height: canvas?.height };
  };
  const capturedViewport = vtkCapture ? viewport() : undefined;
  let capturedBacking: ReturnType<typeof backing> | null = null;
  const view = () => {
    const renderer = lookupVtkRenderWindow(el)?.getRenderer?.() as {
      getActiveCamera?: () => Record<string, (() => unknown) | undefined>;
    } | undefined;
    const camera = renderer?.getActiveCamera?.();
    if (camera) return JSON.stringify(["getPosition", "getFocalPoint", "getViewUp", "getViewAngle", "getParallelProjection", "getParallelScale"].map(name => camera[name]?.()));
    const graph = el as HTMLElement & { _fullLayout?: { xaxis?: { range?: number[] }; yaxis?: { range?: number[] }; scene?: { camera?: unknown } } };
    return JSON.stringify([graph._fullLayout?.xaxis?.range, graph._fullLayout?.yaxis?.range, graph._fullLayout?.scene?.camera]);
  };
  let capturedView = vtkCapture ? view() : undefined;
  const validate = () => {
    if (!el.isConnected || lookupVtkCapture(el) !== vtkCapture) throw new Error("The captured plot was replaced before its image completed");
    if (receipt && !isPlotCanvasReceiptCurrent(receipt)) throw new Error("The plot changed or resized during capture");
    if (capturedViewport !== undefined && viewport() !== capturedViewport) throw new Error("The plot viewport changed during capture");
    if (capturedBacking) {
      const current = backing();
      if (current.canvas !== capturedBacking.canvas || current.width !== capturedBacking.width || current.height !== capturedBacking.height) {
        throw new Error("The plot viewport changed during capture");
      }
    }
    if (capturedView !== undefined && view() !== capturedView) throw new Error("The plot view changed during capture");
    guard.assertCurrent?.();
  };
  const checked = { ...guard, assertCurrent: validate };
  assertCaptureCurrent(checked);
  if (!vtkCapture) {
    receipt = await waitForPlotCanvasReady(el);
    capturedView = view();
    assertCaptureCurrent(checked);
  }
  let releaseQuality = () => {};
  let releaseResolution = () => {};
  try {
    const renderWindow = vtkCapture ? lookupVtkRenderWindow(el) : null;
    if (renderWindow) {
      releaseQuality = holdVtkFullQuality(renderWindow);
      releaseResolution = holdFullVtkPlaybackResolution(renderWindow);
    }
    const uri = await guardedCapture(() => {
      if (!vtkCapture) return Plotly.toImage(el, { format: "png", width: size.width, height: size.height });
      const pending = vtkCapture();
      // Capture setup may restore the saved endpoint's full backing size. Keep
      // its quality holds until every asynchronous image operation is checked.
      capturedBacking = backing();
      return pending;
    }, checked);
    if (!vtkCapture) return uri;
    const image = await dataUriToImageData(uri, size, checked);
    const canvas = document.createElement("canvas"); canvas.width = size.width; canvas.height = size.height;
    const context = canvas.getContext("2d"); if (!context) throw new Error("Capture resize canvas is unavailable");
    context.putImageData(image, 0, 0);
    assertCaptureCurrent(checked);
    return canvas.toDataURL("image/png");
  } finally {
    try { releaseResolution(); } finally { releaseQuality(); }
  }
}

/** Strip the `data:image/png;base64,` prefix from a capture, returning
 *  the bare base64 payload. Tolerates a bare base64 string passing
 *  through unchanged so callers can chain without checking. */
export function dataUriToBase64Png(dataUri: string): string {
  const prefix = "data:image/png;base64,";
  return dataUri.startsWith(prefix) ? dataUri.slice(prefix.length) : dataUri;
}

/** Decode a PNG data URI to an ImageData at the requested size. Used
 *  by the GIF path which needs raw RGBA bytes to quantize. */
export async function dataUriToImageData(
  dataUri: string,
  size: CaptureSize,
  guard: CaptureGuard = {},
): Promise<ImageData> {
  assertCaptureCurrent(guard);
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise<void>((resolve, reject) => {
    const finish = (error?: Error) => {
      img.onload = null; img.onerror = null;
      guard.signal?.removeEventListener("abort", abort);
      if (error) reject(error); else resolve();
    };
    const abort = () => { finish(new DOMException("Capture cancelled", "AbortError")); img.src = ""; };
    img.onload = () => finish();
    img.onerror = () => finish(new Error("PNG decode failed during capture"));
    guard.signal?.addEventListener("abort", abort, { once: true });
    img.src = dataUri;
  });
  assertCaptureCurrent(guard);
  const canvas = document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("2d canvas context unavailable");
  // Fill transparent panels with the active semantic app background so
  // light-theme captures do not acquire dark bands during quantization.
  ctx.fillStyle = getComputedStyle(document.documentElement)
    .getPropertyValue("--color-bg")
    .trim() || "#0e1117";
  ctx.fillRect(0, 0, size.width, size.height);
  const scale = Math.min(size.width / img.naturalWidth, size.height / img.naturalHeight);
  const width = img.naturalWidth * scale, height = img.naturalHeight * scale;
  ctx.drawImage(img, (size.width - width) / 2, (size.height - height) / 2, width, height);
  return ctx.getImageData(0, 0, size.width, size.height);
}
