import { volumeTextureCellCapacity } from "./volumeTexturePacking";

export interface VolumeTextureCapabilities {
  maximumDimension: number;
  maximumCells: number;
  floatLinear: boolean;
}

const capabilities = new WeakMap<WebGL2RenderingContext, VolumeTextureCapabilities>();
const watched = new WeakSet<WebGL2RenderingContext>();

export function volumeTextureCapabilities(gl: WebGL2RenderingContext): VolumeTextureCapabilities {
  if (gl.isContextLost()) {
    capabilities.delete(gl);
    throw new Error("Scientific volume rendering requires an available WebGL2 texture context");
  }
  const previous = capabilities.get(gl);
  if (previous) return previous;
  const maximumDimension = Number(gl.getParameter(gl.MAX_3D_TEXTURE_SIZE));
  const maximumCells = volumeTextureCellCapacity(maximumDimension);
  const current = Object.freeze({ maximumDimension, maximumCells,
    floatLinear: gl.getExtension("OES_texture_float_linear") != null });
  if (!watched.has(gl) && typeof gl.canvas?.addEventListener === "function") {
    const invalidate = () => capabilities.delete(gl);
    gl.canvas.addEventListener("webglcontextlost", invalidate);
    gl.canvas.addEventListener("webglcontextrestored", invalidate);
    watched.add(gl);
  }
  capabilities.set(gl, current);
  return current;
}
