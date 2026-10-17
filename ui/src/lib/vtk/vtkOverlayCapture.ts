import { cssColorbarGradient } from "../plot/plotColormaps";

interface Box { x: number; y: number; width: number; height: number }
interface PaintBox extends Box {
  kind: "box";
  color: string;
  radius: number;
  border: string;
  borderWidth: number;
  gradient: Array<{ color: string; offset: number }> | null;
}
interface PaintText extends Box { kind: "text"; text: string; color: string; font: string; size: number }
interface Overlay { clip: Box; paint: Array<PaintBox | PaintText> }
export interface VtkOverlaySnapshot { width: number; height: number; overlays: Overlay[] }

const PANELS = ".plot3d-colorbar, .plot3d-discrete-legend, .plot3d-badge:not(.plot3d-badge--collapsed), .plot3d-axis-labels";

export function vtkColorbarCaptureStops(name: string): Array<{ color: string; offset: number }> {
  const body = cssColorbarGradient(name).replace(/^linear-gradient\(to top, /, "").slice(0, -1);
  const stops = body.split(/,\s*(?![^()]*\))/);
  return stops.map((stop, index) => {
    const positioned = /^(.*?)\s+([\d.]+)%$/.exec(stop);
    return { color: positioned ? positioned[1] : stop.trim(),
      offset: positioned ? Number(positioned[2]) / 100 : index / Math.max(1, stops.length - 1) };
  });
}

function intersect(a: Box, b: Box): Box {
  const x = Math.max(a.x, b.x), y = Math.max(a.y, b.y);
  return { x, y, width: Math.max(0, Math.min(a.x + a.width, b.x + b.width) - x),
    height: Math.max(0, Math.min(a.y + a.height, b.y + b.height) - y) };
}

// Freeze labels before asynchronous readback, alongside the draw they describe.
export function snapshotVtkOverlays(element: HTMLElement): VtkOverlaySnapshot | null {
  const host = element.closest?.<HTMLElement>(".plot3d-overlay-host");
  if (!host) return null;
  const origin = element.getBoundingClientRect();
  if (!(origin.width > 0 && origin.height > 0)) return null;
  const localBox = (rect: DOMRect): Box => ({ x: rect.x - origin.x, y: rect.y - origin.y,
    width: rect.width, height: rect.height });
  const overlays: Overlay[] = [];
  for (const panel of Array.from(host.querySelectorAll<HTMLElement>(PANELS))) {
    if (panel.closest(".plot3d-overlay-host") !== host) continue;
    let clip: Box = { x: 0, y: 0, width: origin.width, height: origin.height };
    let visible = true;
    for (let parent: HTMLElement | null = panel; parent && parent !== host; parent = parent.parentElement) {
      const style = getComputedStyle(parent);
      if (style.display === "none" || style.visibility === "hidden" || Number(style.opacity) === 0) visible = false;
      if (style.overflow === "hidden" || style.overflow === "clip") clip = intersect(clip, localBox(parent.getBoundingClientRect()));
    }
    if (!visible || clip.width === 0 || clip.height === 0) continue;
    const paint: Overlay["paint"] = [];
    for (const node of [panel, ...Array.from(panel.querySelectorAll<HTMLElement>("*"))]) {
      if (node.closest("button, svg")) continue;
      const style = getComputedStyle(node);
      if (style.display === "none" || style.visibility === "hidden") continue;
      const box = localBox(node.getBoundingClientRect());
      if (box.width === 0 || box.height === 0) continue;
      const colormap = node.dataset.vtkColormap;
      if (style.backgroundColor !== "rgba(0, 0, 0, 0)" || colormap || parseFloat(style.borderWidth) > 0) {
        paint.push({ ...box, kind: "box", color: style.backgroundColor,
          border: style.borderTopColor, borderWidth: parseFloat(style.borderTopWidth) || 0,
          radius: parseFloat(style.borderTopLeftRadius) || 0,
          gradient: colormap ? vtkColorbarCaptureStops(colormap) : null });
      }
      for (const child of Array.from(node.childNodes)) {
        if (child.nodeType !== Node.TEXT_NODE || !child.textContent?.trim()) continue;
        const value = child.textContent;
        const range = document.createRange();
        let line: PaintText | null = null;
        for (let index = 0; index < value.length;) {
          const count = (value.codePointAt(index) ?? 0) > 0xffff ? 2 : 1;
          range.setStart(child, index);
          range.setEnd(child, index + count);
          const rect = localBox(range.getBoundingClientRect());
          const text = value.slice(index, index + count);
          index += count;
          if (rect.width === 0) continue;
          if (line && Math.abs(line.y - rect.y) < 1) {
            line.text += text;
            line.width = rect.x + rect.width - line.x;
          } else {
            line = { ...rect, kind: "text", text, color: style.color,
              font: `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`,
              size: parseFloat(style.fontSize) };
            paint.push(line);
          }
        }
      }
    }
    if (paint.length) overlays.push({ clip, paint });
  }
  return overlays.length ? { width: origin.width, height: origin.height, overlays } : null;
}

export async function compositeVtkOverlays(png: string, snapshot: VtkOverlaySnapshot | null): Promise<string> {
  if (!snapshot) return png;
  const image = new Image();
  await new Promise<void>((resolve, reject) => {
    image.onload = () => resolve();
    image.onerror = () => reject(new Error("Could not decode the captured 3D frame"));
    image.src = png;
  });
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Could not compose the captured 3D legends");
  context.drawImage(image, 0, 0);
  context.scale(canvas.width / snapshot.width, canvas.height / snapshot.height);
  for (const overlay of snapshot.overlays) {
    context.save();
    context.beginPath();
    context.rect(overlay.clip.x, overlay.clip.y, overlay.clip.width, overlay.clip.height);
    context.clip();
    for (const item of overlay.paint) {
      if (item.kind === "text") {
        context.font = item.font;
        context.fillStyle = item.color;
        const metrics = context.measureText(item.text);
        const ascent = metrics.fontBoundingBoxAscent ?? item.size * 0.8;
        const descent = metrics.fontBoundingBoxDescent ?? item.size * 0.2;
        context.fillText(item.text, item.x, item.y + (item.height - ascent - descent) / 2 + ascent);
      } else {
        context.beginPath();
        context.roundRect(item.x, item.y, item.width, item.height, item.radius);
        context.fillStyle = item.color;
        if (item.gradient) {
          const gradient = context.createLinearGradient(0, item.y + item.height, 0, item.y);
          for (const stop of item.gradient) gradient.addColorStop(stop.offset, stop.color);
          context.fillStyle = gradient;
        }
        context.fill();
        if (item.borderWidth > 0) {
          context.strokeStyle = item.border;
          context.lineWidth = item.borderWidth;
          context.stroke();
        }
      }
    }
    context.restore();
  }
  return canvas.toDataURL("image/png");
}
