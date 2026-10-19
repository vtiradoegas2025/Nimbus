interface FloorCommitInput {
  texture: object;
  rgba: Uint8ClampedArray;
  shape: string;
  geometry: string;
}

export class RadarFloorCommitCache {
  private previous: FloorCommitInput | null = null;
  reset(): void { this.previous = null; }
  apply(input: FloorCommitInput, geometry: () => void, data: () => void): void {
    const previous = this.previous;
    if (previous?.texture !== input.texture || previous.geometry !== input.geometry) geometry();
    if (previous?.texture !== input.texture || previous.rgba !== input.rgba || previous.shape !== input.shape) data();
    this.previous = input;
  }
}
