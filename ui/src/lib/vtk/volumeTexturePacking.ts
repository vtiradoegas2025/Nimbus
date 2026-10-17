export interface VolumeTextureLayout {
  sourceShape: [number, number, number];
  textureShape: [number, number, number];
  sourceCells: number;
  textureCells: number;
  tiled: boolean;
}

export function volumeTextureCellCapacity(maximumDimension: number): number {
  if (!Number.isSafeInteger(maximumDimension) || maximumDimension < 2) {
    throw new Error("Scientific volume rendering requires a valid viewer 3D texture limit");
  }
  return Math.min(2_147_483_647, maximumDimension ** 3);
}

export function planVolumeTextureLayout(dimensions: readonly number[], maximumDimension: number): VolumeTextureLayout {
  const capacity = volumeTextureCellCapacity(maximumDimension);
  if (dimensions.length !== 3 || dimensions.some(value => !Number.isSafeInteger(value) || value < 1)) {
    throw new Error("Volume texture dimensions must be positive safe integers");
  }
  const sourceShape = [...dimensions] as [number, number, number];
  const sourceCells = dimensions.reduce((count, value) => count * value, 1);
  if (!Number.isSafeInteger(sourceCells) || sourceCells > capacity) {
    throw new Error(`Scientific volume ${dimensions.join("×")} exceeds the viewer 3D texture limit ${maximumDimension} (${capacity} stored cells)`);
  }
  const tiled = dimensions.some(value => value > maximumDimension);
  const width = tiled ? Math.max(Math.min(dimensions[0], maximumDimension),
    Math.ceil(sourceCells / maximumDimension ** 2)) : dimensions[0];
  const height = tiled ? Math.min(maximumDimension, Math.ceil(sourceCells / width)) : dimensions[1];
  const depth = tiled ? Math.ceil(sourceCells / (width * height)) : dimensions[2];
  const textureCells = width * height * depth;
  if (depth > maximumDimension || textureCells > capacity) {
    throw new Error("Scientific volume texture packing exceeds the viewer capacity");
  }
  return { sourceShape, sourceCells, textureShape: [width, height, depth], textureCells, tiled };
}

export function volumeTextureAddress(index: readonly number[], layout: VolumeTextureLayout): [number, number, number] {
  if (index.length !== 3 || index.some((value, axis) => !Number.isSafeInteger(value)
      || value < 0 || value >= layout.sourceShape[axis])) throw new Error("Volume texture index exceeds its logical lattice");
  const ordinal = index[0] + layout.sourceShape[0] * (index[1] + layout.sourceShape[1] * index[2]);
  const [width, height] = layout.textureShape;
  return [ordinal % width, Math.floor(ordinal / width) % height, Math.floor(ordinal / (width * height))];
}

export function uploadVolumeTexturePrefix(gl: WebGL2RenderingContext, data: Float32Array | Uint8Array,
  layout: VolumeTextureLayout): void {
  if (data.length !== layout.sourceCells) throw new Error("Volume texture source length does not match its logical lattice");
  const [width, height] = layout.textureShape;
  const plane = width * height;
  const slices = Math.floor(data.length / plane);
  const remainder = data.length - slices * plane;
  const rows = Math.floor(remainder / width);
  const tail = remainder % width;
  const type = data instanceof Float32Array ? gl.FLOAT : gl.UNSIGNED_BYTE;
  const parameters = [gl.UNPACK_ALIGNMENT, gl.UNPACK_ROW_LENGTH, gl.UNPACK_IMAGE_HEIGHT,
    gl.UNPACK_SKIP_PIXELS, gl.UNPACK_SKIP_ROWS, gl.UNPACK_SKIP_IMAGES];
  const previous = parameters.map(parameter => gl.getParameter(parameter));
  try {
    parameters.forEach((parameter, index) => gl.pixelStorei(parameter, index === 0 ? 1 : 0));
    if (slices) gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, 0, width, height, slices, gl.RED, type, data, 0);
    if (rows) gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, 0, slices, width, rows, 1, gl.RED, type, data, slices * plane);
    if (tail) gl.texSubImage3D(gl.TEXTURE_3D, 0, 0, rows, slices, tail, 1, 1, gl.RED, type, data, slices * plane + rows * width);
  } finally {
    parameters.forEach((parameter, index) => gl.pixelStorei(parameter, previous[index]));
  }
}

export const VOLUME_TEXTURE_ADDRESS_GLSL = `
uniform vec3 nimbusSourceDimensions[4];
uniform vec4 nimbusTiledTexture;
ivec3 nimbusTextureAddress(ivec3 index, ivec3 source, ivec3 packed) {
  int ordinal = index.x + source.x * (index.y + source.y * index.z);
  return ivec3(ordinal % packed.x, (ordinal / packed.x) % packed.y, ordinal / (packed.x * packed.y));
}
vec4 nimbusLogicalTexel(highp sampler3D image, ivec3 index, ivec3 source, int component) {
  return texelFetch(image, nimbusTiledTexture[component] > 0.5
    ? nimbusTextureAddress(index, source, textureSize(image, 0)) : index, 0);
}
`;
