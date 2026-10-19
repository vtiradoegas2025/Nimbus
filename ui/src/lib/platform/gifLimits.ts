export const GIF_MAX_BYTES = 32 * 1024 * 1024;
export const GIF_MAX_FRAMES = 2000;
export const GIF_MAX_PIXELS = 4096 * 2160;

export function validateGifSize(width: number, height: number): void {
  if (![width, height].every(value => Number.isSafeInteger(value) && value > 0 && value <= 65535)
    || width * height > GIF_MAX_PIXELS) {
    throw new Error("GIF dimensions exceed the 4096 × 2160 pixel budget; export a smaller figure or PNG");
  }
}
