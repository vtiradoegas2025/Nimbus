/* Frame-capture registry for VTK widgets. captureNextImage arms a one-shot
 * grab that resolves only on the next render pass — always arm then render. */

import {
  collectVtkAppearanceEvidence,
  type VtkAppearanceDescriptor,
  type VtkAppearanceEvidence,
  type VtkAppearanceRenderWindowLike,
} from "./vtkAppearanceEvidence";
import { holdFullVtkPlaybackResolution } from "./vtkPlaybackScreenResolution";
import { compositeVtkOverlays, snapshotVtkOverlays } from "./vtkOverlayCapture";
import { holdVtkFullQuality } from "./vtkInteraction";

export type FrameCapture = () => Promise<string>;

export interface VtkAppearanceCapture {
  pngDataUri: string;
  evidence: VtkAppearanceEvidence;
}

export type VtkAppearanceEvidenceCapture = () => Promise<VtkAppearanceCapture>;

/** Minimal shape of a vtkGenericRenderWindow needed to capture a PNG. Typed
 *  structurally so this module stays free of vtk.js imports. */
interface VtkGenericRenderWindowLike {
  getApiSpecificRenderWindow: () => {
    captureNextImage?: (format: string) => Promise<string> | string | undefined;
  };
  getRenderWindow: () => { render: () => void };
}

/**
 * Capture the current 3D scene to a PNG data URL. Arms `captureNextImage`,
 * then issues a render so the capture pass actually runs and resolves the
 * promise; a timeout guard turns a future regression into an error instead of
 * an unkillable hang in the recorder loop.
 */
export async function captureVtkPng(
  renderWindow: VtkGenericRenderWindowLike,
  timeoutMs = 8000,
): Promise<string> {
  const releaseResolution = holdFullVtkPlaybackResolution(renderWindow);
  try {
    const glWindow = renderWindow.getApiSpecificRenderWindow();
    const pending = Promise.resolve(glWindow.captureNextImage?.("image/png"));
    // On the timeout path `Promise.race` settles via `guard`, leaving `pending`
    // unsettled; a later rejection would surface as an unhandledrejection. Attach
    // a second no-op handler so a post-timeout rejection is always handled. The
    // race still observes `pending` directly, so the normal-path resolution and
    // error both propagate below.
    pending.catch(() => undefined);
    // Must render AFTER arming so the pass delivers the image.
    renderWindow.getRenderWindow().render();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const guard = new Promise<never>((_, reject) => {
      timer = setTimeout(
        () => reject(new Error("VTK frame capture timed out")),
        timeoutMs,
      );
    });
    try {
      const value = await Promise.race([pending, guard]);
      if (typeof value !== "string") {
        throw new Error("VTK captureNextImage returned no data");
      }
      return value;
    } finally {
      if (timer) clearTimeout(timer);
    }
  } finally { releaseResolution(); }
}

// WeakMap: a missed unregister would otherwise pin an entire GPU scene for the process lifetime.
interface VtkCaptureRegistration {
  frame: FrameCapture;
  appearance: VtkAppearanceEvidenceCapture | null;
  renderWindow?: VtkAppearanceRenderWindowLike;
  descriptor?: () => VtkAppearanceDescriptor;
}

const registry = new WeakMap<HTMLElement, VtkCaptureRegistration>();

export function registerVtkCapture(
  el: HTMLElement,
  capture: FrameCapture,
  appearance: VtkAppearanceEvidenceCapture | null = null,
): void {
  registry.set(el, { frame: capture, appearance });
}

export function registerVtkAppearanceCapture(
  el: HTMLElement,
  renderWindow: VtkAppearanceRenderWindowLike,
  descriptor: () => VtkAppearanceDescriptor,
  beforeFrameCapture?: () => void,
): void {
  const frame = async () => {
    const releaseQuality = holdVtkFullQuality(renderWindow);
    try {
      beforeFrameCapture?.();
      const overlays = snapshotVtkOverlays(el);
      return await compositeVtkOverlays(await captureVtkPng(renderWindow), overlays);
    } finally { releaseQuality(); }
  };
  registerVtkCapture(el, frame, async () => {
    const releaseResolution = holdFullVtkPlaybackResolution(renderWindow);
    const releaseQuality = holdVtkFullQuality(renderWindow);
    try {
      const evidence = collectVtkAppearanceEvidence(renderWindow, descriptor());
      const pngDataUri = await captureVtkPng(renderWindow);
      return { pngDataUri, evidence };
    } finally { releaseQuality(); releaseResolution(); }
  });
  registry.get(el)!.renderWindow = renderWindow;
  registry.get(el)!.descriptor = descriptor;
}

export function lookupVtkRenderWindow(el: HTMLElement | null): VtkAppearanceRenderWindowLike | null {
  while (el) {
    const renderWindow = registry.get(el)?.renderWindow;
    if (renderWindow) return renderWindow;
    el = el.parentElement;
  }
  return null;
}

export function unregisterVtkCapture(el: HTMLElement): void {
  registry.delete(el);
}

export function lookupVtkCapture(el: HTMLElement | null): FrameCapture | null {
  if (!el) return null;
  // Walk up in case the recorder received a wrapping div rather than the exact registered element.
  let current: HTMLElement | null = el;
  while (current) {
    const registration = registry.get(current);
    if (registration) return registration.frame;
    current = current.parentElement;
  }
  return null;
}

export function lookupVtkAppearanceCapture(
  el: HTMLElement | null,
): VtkAppearanceEvidenceCapture | null {
  let current = el;
  while (current) {
    const registration = registry.get(current);
    if (registration?.appearance) return registration.appearance;
    current = current.parentElement;
  }
  return null;
}

export function lookupVtkDescriptor(el: HTMLElement | null): (() => VtkAppearanceDescriptor) | null {
  while (el) {
    const descriptor = registry.get(el)?.descriptor;
    if (descriptor) return descriptor;
    el = el.parentElement;
  }
  return null;
}
