import { waitForCapturePaint } from "@/lib/run/capturePaint";
import { captureFrameDataUri, type CaptureSize } from "@/lib/run/frameCapture";
import { assertCaptureCurrent, type CaptureGuard } from "@/lib/run/captureGuard";
import { buildPortableViewpoints, orbitPoint, PORTABLE_VIEWPOINT_COUNTS, PORTABLE_VIEWPOINT_MAX_BYTES, type PortableViewpoint } from "@/lib/platform/portableViewpoints";
import { lookupVtkRenderWindow, lookupVtkDescriptor } from "./vtkCaptureRegistry";
import { holdVtkFullQuality } from "./vtkInteraction";
import { holdFullVtkPlaybackResolution } from "./vtkPlaybackScreenResolution";
import { collectVtkAppearanceEvidence } from "./vtkAppearanceEvidence";
import { holdVtkSceneRendering } from "./vtkContextRenderGuard";

interface Camera {
  getPosition: () => number[]; getFocalPoint: () => number[]; getViewUp: () => number[];
  setPosition: (...p: number[]) => void; setViewUp: (...p: number[]) => void;
  getClippingRange: () => number[]; setClippingRange: (...p: number[]) => void;
}
export async function captureViewpoints(element: HTMLElement, size: CaptureSize, count: number, title: string, source: unknown,
  guard: CaptureGuard, onProgress: (done: number) => void): Promise<Uint8Array> {
  if (!(PORTABLE_VIEWPOINT_COUNTS as readonly number[]).includes(count)) throw new Error("Invalid viewpoint count");
  if (size.width * size.height > 16_777_216) throw new Error("Reduce the viewport below 16 million pixels before exporting viewpoints");
  const window = lookupVtkRenderWindow(element);
  const descriptor = lookupVtkDescriptor(element);
  if (!window || !descriptor) throw new Error("Interactive viewpoints require a complete 3D plot");
  const renderer = window.getRenderer() as { getActiveCamera: () => Camera; resetCameraClippingRange: () => void };
  const camera = renderer.getActiveCamera();
  const position = [...camera.getPosition()], center = [...camera.getFocalPoint()], up = [...camera.getViewUp()], clip = [...camera.getClippingRange()];
  const releaseQuality = holdVtkFullQuality(window), releaseResolution = holdFullVtkPlaybackResolution(window);
  const dimensions = () => {
    const rect = element.getBoundingClientRect(), canvas = element.querySelector("canvas");
    return JSON.stringify([rect.width, rect.height, canvas?.width, canvas?.height, element.ownerDocument.defaultView?.devicePixelRatio]);
  };
  const initialDimensions = dimensions();
  const current: CaptureGuard = { ...guard, assertCurrent: () => {
    guard.assertCurrent?.();
    if (lookupVtkRenderWindow(element) !== window || dimensions() !== initialDimensions) throw new Error("The scene or viewport changed during viewpoint export");
  } };
  const views: PortableViewpoint[] = [];
  let bytes = 0;
  try {
    const scene = collectVtkAppearanceEvidence(window, descriptor());
    for (let i = 0; i < count; i++) {
      assertCaptureCurrent(current);
      if (lookupVtkRenderWindow(element) !== window) throw new Error("3D scene changed during viewpoint export");
      const angleDeg = i * 360 / count;
      const releaseTraversal = holdVtkSceneRendering(window);
      try {
        camera.setPosition(...orbitPoint(position, center, angleDeg));
        camera.setViewUp(...orbitPoint(up, [0, 0, 0], angleDeg));
        renderer.resetCameraClippingRange();
        // Camera subscriptions update ticks and labels before the capture's
        // full-quality draw; their incidental renders must not duplicate it.
        await waitForCapturePaint(guard.signal);
      } finally { releaseTraversal(); }
      assertCaptureCurrent(current);
      const png = await captureFrameDataUri(element, size, current);
      bytes += png.length;
      if (bytes > PORTABLE_VIEWPOINT_MAX_BYTES - 256 * 1024) throw new Error("Portable viewer exceeds 64 MiB; choose fewer viewpoints or a smaller viewport");
      views.push({ angleDeg, png }); onProgress(i + 1);
    }
    assertCaptureCurrent(current);
    return buildPortableViewpoints({ schema: "nimbus.portable-viewpoints", version: 1, title, width: size.width, height: size.height, source, scene, views });
  } finally {
    try {
      if (lookupVtkRenderWindow(element) === window) {
        camera.setPosition(...position); camera.setViewUp(...up); camera.setClippingRange(...clip);
        window.getRenderWindow().render();
      }
    } finally { try { releaseResolution(); } finally { releaseQuality(); } }
  }
}
