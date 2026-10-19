import assert from "node:assert/strict";
import { volumeSamplingShader } from "../../../../ui/src/lib/vtk/volumeSamplingShader.ts";
import { planVolumeTextureLayout } from "../../../../ui/src/lib/vtk/volumeTexturePacking.ts";

const native = volumeSamplingShader([planVolumeTextureLayout([8, 7, 6], 16)]);
const hardware = volumeSamplingShader([planVolumeTextureLayout([8, 7, 6], 16)], [false]);
assert.ok(!hardware.scalar.includes("texelFetch") && !hardware.scalar.includes("ordinal") && !hardware.scalar.includes("%"));
assert.ok(!hardware.scalar.includes("nimbusManualScalarFilter"));
assert.notEqual(native.key, hardware.key);
assert.ok(!native.scalar.includes("ordinal") && !native.scalar.includes("%") && !native.scalar.includes("nimbusPacked"));
assert.ok(!native.scalar.includes("nimbusTiledTexture") && !native.scalar.includes("nimbusSourceDimensions"));
assert.equal((native.scalar.match(/texelFetch\(/g) ?? []).length, 8);
assert.ok(native.scalar.includes("return texture(image, coordinate)"));
assert.ok(native.validity.includes("texture(nimbusValidityTexture[0], pos)"));
assert.equal(native.key, volumeSamplingShader([planVolumeTextureLayout([7, 6, 5], 16)]).key);

const layout = planVolumeTextureLayout([2, 3, 41], 8);
const packed = volumeSamplingShader([layout, planVolumeTextureLayout([4, 4, 4], 8)]);
assert.ok(packed.scalar.includes("index.x + 2 * (index.y + 3 * index.z)"));
assert.ok(packed.scalar.includes(`ordinal % ${layout.textureShape[0]}`));
assert.ok(packed.scalar.includes(`ordinal / ${layout.textureShape[0] * layout.textureShape[1]}`));
assert.ok(packed.validity.includes("nimbusPackedSample0(nimbusValidityTexture[0], pos)"));
assert.ok(!packed.validity.includes("nimbusPackedScalar0"));
assert.ok(packed.validity.includes("texture(nimbusValidityTexture[1], pos)"));
assert.notEqual(packed.key, native.key);
assert.notEqual(packed.key, volumeSamplingShader([planVolumeTextureLayout([3, 2, 41], 8)]).key);
const packedLinear = volumeSamplingShader([layout], [false], { validity: [false], nearest: [false] });
assert.ok(!packedLinear.scalar.includes("nimbusNearestScalarFilter"));
assert.ok(!packedLinear.validity.includes("sampler3D") && !packedLinear.validity.includes("volume.inverseDimensions"));
assert.ok(packedLinear.scalar.includes("nimbusPackedSample0"));
const packedNearest = volumeSamplingShader([layout], [false], { validity: [false], nearest: [true] });
assert.ok(!packedNearest.scalar.includes("nimbusNearestScalarFilter") && !packedNearest.scalar.includes("nimbusPackedSample0"));
assert.ok(!packedNearest.scalar.includes("mix("));
assert.notEqual(packedNearest.key, packedLinear.key);
const maskedNearest = volumeSamplingShader([layout], [false], { validity: [true], nearest: [true] });
assert.ok(maskedNearest.scalar.includes("nimbusPackedSample0"));
assert.ok(maskedNearest.validity.includes("nimbusPackedSample0(nimbusValidityTexture[0], pos)"));
assert.ok(maskedNearest.validity.includes("nimbusValid0(pos + delta)") && maskedNearest.validity.includes("nimbusValid0(pos - delta)"));
assert.notEqual(maskedNearest.key, packedNearest.key);

const mixed = [layout, planVolumeTextureLayout([4, 4, 4], 8), planVolumeTextureLayout([3, 2, 41], 8), null];
for (let mask = 0; mask < 16; mask++) for (let filter = 0; filter < 16; filter++) {
  const validity = mixed.map((_, index) => Boolean(mask & (1 << index)));
  const nearest = mixed.map((_, index) => Boolean(filter & (1 << index)));
  const shader = volumeSamplingShader(mixed, [false, false, false, false], { validity, nearest });
  assert.ok(!shader.scalar.includes("nimbusNearestScalarFilter"));
  for (let index = 0; index < 4; index++) {
    const body = shader.validity.match(new RegExp(`bool nimbusValid${index}\\(vec3 pos\\) \\{([^}]+)`))![1];
    assert.equal(body.includes("return true;") && !body.includes("nimbusValidityEnabled"), !validity[index]);
    if (validity[index]) assert.ok(body.includes(`nimbusValidityTexture[${index}]`));
  }
  assert.equal(shader.validity.includes("volume.inverseDimensions"), validity[0]);
  assert.notEqual(shader.key, volumeSamplingShader(mixed, [false, false, false, false], {
    validity: validity.map((value, index) => index === 0 ? !value : value), nearest,
  }).key);
}
console.log("256 mask/filter combinations specialize actual inputs, preserving exact linear mask support and rebuilding when validity or interpolation changes");
