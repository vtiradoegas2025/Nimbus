import assert from "node:assert/strict";
import { planVolumeTextureLayout, uploadVolumeTexturePrefix, volumeTextureAddress,
  volumeTextureCellCapacity } from "../../../../ui/src/lib/vtk/volumeTexturePacking.ts";

let cases = 0;
for (const maximum of [2, 3, 4, 7, 16, 32]) {
  for (const shape of [[2, 2, maximum + 1], [maximum + 1, 2, 2], [2, maximum + 1, 2],
    [3, 5, 11], [2, 3, maximum * maximum - 1], [maximum, maximum, maximum]]) {
    const cells = shape.reduce((total, count) => total * count, 1);
    if (cells > volumeTextureCellCapacity(maximum)) {
      assert.throws(() => planVolumeTextureLayout(shape, maximum), /viewer 3D texture limit/); cases++; continue;
    }
    const layout = planVolumeTextureLayout(shape, maximum);
    assert.ok(layout.textureShape.every(count => count <= maximum));
    assert.ok(layout.textureCells >= cells);
    for (let z = 0; z < shape[2]; z++) for (let y = 0; y < shape[1]; y++) for (let x = 0; x < shape[0]; x++) {
      const address = volumeTextureAddress([x, y, z], layout);
      assert.ok(address.every((value, axis) => value >= 0 && value < layout.textureShape[axis]));
      const stored = address[0] + layout.textureShape[0] * (address[1] + layout.textureShape[1] * address[2]);
      assert.equal(stored, x + shape[0] * (y + shape[1] * z));
      cases++;
    }
    for (const source of [Float32Array.from({ length: cells }, (_, i) => Math.fround((i % 19) - 8.125)),
      Uint8Array.from({ length: cells }, (_, i) => i % 5 === 0 ? 0 : 255)]) {
      const storage = source instanceof Float32Array ? new Float32Array(layout.textureCells) : new Uint8Array(layout.textureCells);
      const original = source.slice();
      let posts = 0;
      const settings = new Map<number, number>([[1, 8], [2, 17], [3, 19], [4, 2], [5, 3], [6, 4]]);
      const before = [...settings];
      const gl = {
        UNPACK_ALIGNMENT: 1, UNPACK_ROW_LENGTH: 2, UNPACK_IMAGE_HEIGHT: 3,
        UNPACK_SKIP_PIXELS: 4, UNPACK_SKIP_ROWS: 5, UNPACK_SKIP_IMAGES: 6,
        FLOAT: 7, UNSIGNED_BYTE: 8, TEXTURE_3D: 9, RED: 10,
        getParameter: (key: number) => settings.get(key),
        pixelStorei: (key: number, value: number) => settings.set(key, value),
        texSubImage3D: (_target: number, _level: number, tx: number, ty: number, tz: number,
          width: number, height: number, depth: number, _format: number, type: number,
          values: Float32Array | Uint8Array, offset: number) => {
          assert.equal(values, source);
          assert.equal(type, source instanceof Float32Array ? 7 : 8);
          assert.equal(settings.get(1), 1);
          for (const key of [2, 3, 4, 5, 6]) assert.equal(settings.get(key), 0);
          posts++;
          let i = offset;
          for (let z = 0; z < depth; z++) for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
            const ordinal = tx + x + layout.textureShape[0] * (ty + y + layout.textureShape[1] * (tz + z));
            storage[ordinal] = values[i++];
          }
        },
      } as unknown as WebGL2RenderingContext;
      uploadVolumeTexturePrefix(gl, source, layout);
      assert.deepEqual(source, original);
      assert.deepEqual(storage.slice(0, cells), source);
      assert.ok(storage.slice(cells).every(value => value === 0));
      assert.ok(posts > 0 && posts <= 3);
      assert.deepEqual([...settings], before);
      cases++;
    }
  }
}
for (const dimensions of [[0, 2, 2], [2.5, 2, 2], [Infinity, 2, 2], [2, 2]]) {
  assert.throws(() => planVolumeTextureLayout(dimensions, 16), /positive safe integers/);
}
assert.throws(() => volumeTextureAddress([2, 0, 0], planVolumeTextureLayout([2, 2, 3], 4)), /logical lattice/);
assert.throws(() => uploadVolumeTexturePrefix({} as WebGL2RenderingContext, new Float32Array(2),
  planVolumeTextureLayout([2, 2, 3], 4)), /source length/);
console.log(`Exact scalar/mask GPU packing: ${cases} cases passed; immutable source prefix uploads use at most3 slabs`);
