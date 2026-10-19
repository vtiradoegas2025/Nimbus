declare module "gifenc" {
  export interface GifEncoderHandle {
    writeFrame(
      index: Uint8Array,
      width: number,
      height: number,
      opts?: {
        palette?: number[][];
        delay?: number;
        transparent?: boolean;
        transparentIndex?: number;
        dispose?: number;
        first?: boolean;
        repeat?: number;
      },
    ): void;
    finish(): void;
    bytes(): Uint8Array;
    reset(): void;
    bytesView(): Uint8Array;
  }

  export function GIFEncoder(opts?: { auto?: boolean; initialCapacity?: number }): GifEncoderHandle;

  export function quantize(
    rgba: Uint8Array | Uint8ClampedArray,
    maxColors: number,
    opts?: {
      format?: "rgba4444" | "rgba565" | "rgb444" | "rgb565";
      oneBitAlpha?: boolean | number;
      clearAlpha?: boolean;
      clearAlphaThreshold?: number;
      clearAlphaColor?: number;
    },
  ): number[][];

  export function applyPalette(
    rgba: Uint8Array | Uint8ClampedArray,
    palette: number[][],
    format?: "rgba4444" | "rgba565" | "rgb444" | "rgb565",
  ): Uint8Array;

  export function nearestColorIndex(palette: number[][], pixel: number[]): number;
}
