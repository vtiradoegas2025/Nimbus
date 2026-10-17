import { GIFEncoder, quantize, applyPalette } from "gifenc";
import { gifFrameDelayMs } from "@/lib/run/captureGuard";
import { validateGifSize, GIF_MAX_BYTES, GIF_MAX_FRAMES } from "./gifLimits";

export class GifEncoding {
  private encoder = GIFEncoder();
  private palette: number[][] | null = null;
  private count = 0;
  private finished = false;

  private width: number;
  private height: number;
  private fps: number;
  private maxBytes: number;

  constructor(width: number, height: number, fps: number, maxBytes = GIF_MAX_BYTES) {
    this.width = width; this.height = height; this.fps = fps; this.maxBytes = maxBytes;
    validateGifSize(width, height);
    gifFrameDelayMs(0, fps);
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > GIF_MAX_BYTES) throw new Error("Invalid GIF byte limit");
  }

  append(rgba: Uint8ClampedArray): void {
    if (this.finished) throw new Error("GIF is already finished");
    if (rgba.length !== this.width * this.height * 4) throw new Error("GIF frame dimensions changed");
    if (this.count >= GIF_MAX_FRAMES) throw new Error("GIF frame limit reached; choose fewer source frames or MP4");
    // Reserve more than the maximum 12-bit LZW codes, block headers and palette
    // before encoding, so even incompressible images cannot overrun the limit.
    if (this.encoder.bytesView().length + this.width * this.height * 3 + 2048 > this.maxBytes) {
      throw new Error("GIF memory limit reached; choose a shorter range, fewer source frames, or MP4");
    }
    this.palette ??= buildAnimationPalette(rgba);
    this.encoder.writeFrame(applyPalette(rgba, this.palette), this.width, this.height, {
      palette: this.count === 0 ? this.palette : undefined,
      delay: gifFrameDelayMs(this.count, this.fps),
    });
    this.count++;
  }

  finish(): Uint8Array {
    if (this.finished || !this.count) throw new Error("GIF has no unfinished frames");
    this.finished = true;
    this.encoder.finish();
    return this.encoder.bytesView();
  }
}

const RESERVED_SCIENTIFIC_COLORS: readonly (readonly [number, number, number])[] = [
  // Neutral scene anchors.
  [0, 0, 0], [10, 13, 17], [24, 28, 34], [64, 68, 74],
  [112, 116, 122], [160, 164, 170], [208, 212, 218], [255, 255, 255],
  // Flow, vortex, and condensate diagnostics.
  [217, 163, 64], [92, 156, 235], [235, 79, 92], [151, 98, 213],
  [72, 169, 164], [199, 214, 230], [53, 168, 212], [110, 196, 222],
  // Radar-like progression, including severe/extreme bins.
  [4, 233, 231], [1, 160, 246], [0, 0, 246], [0, 255, 0],
  [0, 200, 0], [0, 144, 0], [255, 255, 0], [231, 192, 0],
  [255, 144, 0], [255, 0, 0], [214, 0, 0], [192, 0, 0],
  [255, 0, 255], [153, 85, 201], [255, 204, 255], [118, 72, 42],
];

/** Build one palette for the entire animation. Most entries preserve the
 * captured scene's actual tonal range; the reserved tail protects scientific
 * overlays that may be absent from an early source frame. */
export function buildAnimationPalette(
  rgba: Uint8Array | Uint8ClampedArray,
): number[][] {
  const sceneColors = quantize(rgba, 256 - RESERVED_SCIENTIFIC_COLORS.length);
  return [
    ...sceneColors,
    ...RESERVED_SCIENTIFIC_COLORS.map((color) => [...color]),
  ];
}
