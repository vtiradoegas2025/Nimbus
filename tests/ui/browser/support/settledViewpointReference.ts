import { captureFrameDataUri } from "@nimbus/lib/run/frameCapture";
import { waitForCapturePaint } from "@nimbus/lib/run/capturePaint";
import { orbitPoint } from "@nimbus/lib/platform/portableViewpoints";
import { lookupVtkRenderWindow } from "@nimbus/lib/vtk/vtkCaptureRegistry";
import { snapshotVtkOverlays } from "@nimbus/lib/vtk/vtkOverlayCapture";
import { holdVtkFullQuality } from "@nimbus/lib/vtk/vtkInteraction";
import { holdFullVtkPlaybackResolution } from "@nimbus/lib/vtk/vtkPlaybackScreenResolution";

export async function captureSettledViewpointReferences(element: HTMLElement, angles: readonly number[]) {
  if (angles.length > 12 || angles.some(angle => !Number.isFinite(angle))) throw new Error("Invalid reference viewpoints");
  const window = lookupVtkRenderWindow(element);
  const canvas = element.querySelector("canvas");
  if (!window || !canvas) throw new Error("Reference requires a mounted VTK scene");
  const renderer = window.getRenderer() as {
    getActiveCamera(): {
      getPosition(): number[]; getFocalPoint(): number[]; getViewUp(): number[]; getClippingRange(): number[];
      setPosition(...values: number[]): void; setViewUp(...values: number[]): void; setClippingRange(...values: number[]): void;
    };
    resetCameraClippingRange(): void;
  };
  const camera = renderer.getActiveCamera();
  const position = [...camera.getPosition()], center = [...camera.getFocalPoint()];
  const up = [...camera.getViewUp()], clip = [...camera.getClippingRange()];
  const releaseQuality = holdVtkFullQuality(window), releaseResolution = holdFullVtkPlaybackResolution(window);
  const references: { angleDeg: number; png: string; overlays: ReturnType<typeof snapshotVtkOverlays> }[] = [];
  try {
    for (const angleDeg of angles) {
      camera.setPosition(...orbitPoint(position, center, angleDeg));
      camera.setViewUp(...orbitPoint(up, [0, 0, 0], angleDeg));
      renderer.resetCameraClippingRange();
      window.getRenderWindow().render();
      let previous: string | null = null, stable = 0;
      for (let paint = 0; paint < 10 && stable < 2; paint++) {
        await waitForCapturePaint();
        const current = JSON.stringify(snapshotVtkOverlays(element));
        stable = current === previous ? stable + 1 : 0;
        previous = current;
      }
      if (stable < 2) throw new Error("Reference overlays did not stabilize at the requested camera");
      const overlays = snapshotVtkOverlays(element);
      const png = await captureFrameDataUri(element, { width: canvas.width, height: canvas.height });
      if (JSON.stringify(snapshotVtkOverlays(element)) !== JSON.stringify(overlays)) {
        throw new Error("Reference annotations changed during PNG capture");
      }
      references.push({ angleDeg, png, overlays });
    }
    return references;
  } finally {
    try {
      camera.setPosition(...position); camera.setViewUp(...up); camera.setClippingRange(...clip);
      window.getRenderWindow().render();
    } finally { try { releaseResolution(); } finally { releaseQuality(); } }
  }
}
