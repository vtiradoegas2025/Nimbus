export interface OwnedCanvasCaptureRequest {
  promise: Promise<string>;
  cancel(): void;
}

interface CanvasViewProp {
  getContainer?: () => HTMLElement;
}

interface CanvasRenderer {
  getViewProps(): CanvasViewProp[];
}

export function captureOwnedCanvasImage(
  source: HTMLCanvasElement,
  renderers: CanvasRenderer[],
  format = "image/png",
): string {
  const canvas = document.createElement("canvas");
  canvas.width = source.width;
  canvas.height = source.height;
  try {
    const context = canvas.getContext("2d");
    if (!context) throw new Error("VTK snapshot canvas is unavailable");
    context.drawImage(source, 0, 0);
    const origin = source.getBoundingClientRect();
    for (const renderer of renderers) {
      for (const prop of renderer.getViewProps()) {
        if (!prop.getContainer) continue;
        const children = prop.getContainer().getElementsByTagName("canvas");
        for (const child of Array.from(children)) {
          const bounds = child.getBoundingClientRect();
          context.drawImage(child, bounds.x - origin.x, bounds.y - origin.y);
        }
      }
    }
    return canvas.toDataURL(format);
  } finally {
    canvas.width = 0;
    canvas.height = 0;
    canvas.remove();
  }
}
