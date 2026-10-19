import type { StreamlineExtractionResult, StreamlineGeometry } from "./streamlineExtractionProtocol";

function hashBytes(hash: number, view: ArrayBufferView): number {
  const bytes = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
  let value = hash >>> 0;
  for (const byte of bytes) {
    value ^= byte;
    value = Math.imul(value, 0x01000193) >>> 0;
  }
  return value;
}

function hashGeometry(hash: number, geometry: StreamlineGeometry | null): number {
  if (!geometry) return Math.imul(hash ^ 0xff, 0x01000193) >>> 0;
  let value = hashBytes(hash, geometry.points);
  value = hashBytes(value, geometry.lines);
  value = hashBytes(value, geometry.polys);
  value = hashBytes(value, geometry.strips);
  value = hashBytes(value, geometry.normals);
  if (geometry.scalars) value = hashBytes(value, geometry.scalars);
  return value;
}

export function streamlineGeometryFingerprint(result: StreamlineExtractionResult | null): string | null {
  if (!result) return null;
  let hash = 0x811c9dc5;
  hash = hashGeometry(hash, result.forward.path);
  hash = hashGeometry(hash, result.backward.path);
  hash = hashGeometry(hash, result.forward.arrows);
  hash = hashGeometry(hash, result.backward.arrows);
  return `${result.integratedPoints}:${hash.toString(16).padStart(8, "0")}`;
}

export function streamlineSeedFingerprint(seeds: Float32Array | null): string | null {
  if (!seeds) return null;
  return `${seeds.length / 3}:${hashBytes(0x811c9dc5, seeds).toString(16).padStart(8, "0")}`;
}
