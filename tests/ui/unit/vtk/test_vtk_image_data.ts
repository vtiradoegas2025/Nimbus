import assert from "node:assert/strict";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import {
  setPreparedImageDisplayGeometry,
  setPreparedImageScalars,
  setPreparedImageVectors,
} from "../../../../ui/src/lib/vtk/vtkImageData.ts";
import { getVolumeValidity } from "../../../../ui/src/lib/vtk/volumeValidity.ts";

const image = vtkImageData.newInstance();
const values = Float32Array.from({ length: 24 }, (_, index) => index - 12);
const prepared = {
  shape: [2, 3, 4] as [number, number, number],
  spacing: [2, 3, 4] as [number, number, number],
  origin: [-20, 30, 40] as [number, number, number],
  values,
};
setPreparedImageScalars(image, prepared);
const scalars = image.getPointData().getScalars();
const scalarTime = scalars.getMTime();
const imageTime = image.getMTime();
setPreparedImageScalars(image, prepared);
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(scalars.getMTime(), scalarTime);
assert.equal(image.getMTime(), imageTime);
setPreparedImageScalars(image, { ...prepared, values: new Float32Array(values.buffer, values.byteOffset, values.length) });
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(image.getMTime(), imageTime);

setPreparedImageScalars(image, { ...prepared, spacing: [4, 6, 8], origin: [-40, 60, 80] });
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(scalars.getMTime(), scalarTime);
assert.deepEqual(image.getSpacing(), [4, 6, 8]);
assert.deepEqual(image.getOrigin(), [-40, 60, 80]);
assert.deepEqual([...scalars.getData()], [...values]);

const mask = new Uint8Array(values.length).fill(255);
mask[3] = 0;
setPreparedImageScalars(image, { ...prepared, validity: mask });
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(getVolumeValidity(image), mask);
const nextMask = mask.slice();
nextMask[5] = 0;
setPreparedImageScalars(image, { ...prepared, validity: nextMask });
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(getVolumeValidity(image), nextMask);
setPreparedImageScalars(image, prepared);
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(getVolumeValidity(image), undefined);

const beforeInvalid = { spacing: [...image.getSpacing()], origin: [...image.getOrigin()], shape: [...image.getDimensions()], time: image.getMTime() };
for (const invalid of [
  { ...prepared, shape: [0.5, 6, 8] as [number, number, number] },
  { ...prepared, shape: [0, 3, 4] as [number, number, number] },
  { ...prepared, spacing: [2, NaN, 4] as [number, number, number] },
  { ...prepared, spacing: [2, -3, 4] as [number, number, number] },
  { ...prepared, origin: [-20, Infinity, 40] as [number, number, number] },
  { ...prepared, values: new Float32Array(23) },
  { ...prepared, validity: new Uint8Array(23) },
]) {
  assert.throws(() => setPreparedImageScalars(image, invalid));
  assert.equal(image.getPointData().getScalars(), scalars);
  assert.deepEqual({ spacing: [...image.getSpacing()], origin: [...image.getOrigin()], shape: [...image.getDimensions()], time: image.getMTime() }, beforeInvalid);
}
for (const scale of [{ xy: 1, z: NaN }, { xy: 0, z: 1 }, { xy: 1, z: Infinity }, { xy: -1, z: 1 }]) {
  assert.throws(() => setPreparedImageDisplayGeometry(image, prepared, scale));
  assert.deepEqual({ spacing: [...image.getSpacing()], origin: [...image.getOrigin()], shape: [...image.getDimensions()], time: image.getMTime() }, beforeInvalid);
}
assert.throws(() => setPreparedImageDisplayGeometry(image, { spacing: [1e308, 1, 1], origin: [0, 0, 0] }, { xy: 100, z: 1 }));
assert.deepEqual({ spacing: [...image.getSpacing()], origin: [...image.getOrigin()], shape: [...image.getDimensions()], time: image.getMTime() }, beforeInvalid);

setPreparedImageDisplayGeometry(image, prepared, { xy: 1, z: 2 });
assert.equal(image.getPointData().getScalars(), scalars);
assert.equal(scalars.getMTime(), scalarTime);
assert.deepEqual(image.getSpacing(), [2, 3, 8]);
assert.deepEqual(image.getOrigin(), [-20, 30, 80]);
const scaledTime = image.getMTime();
setPreparedImageDisplayGeometry(image, prepared, { xy: 1, z: 2 });
assert.equal(image.getMTime(), scaledTime);

setPreparedImageScalars(image, { ...prepared, shape: [4, 3, 2] });
assert.notEqual(image.getPointData().getScalars(), scalars);
assert.deepEqual(image.getDimensions(), [4, 3, 2]);
assert.deepEqual([...image.getPointData().getScalars().getData()], [...values]);
const reshaped = image.getPointData().getScalars();
const changed = Float32Array.from(values, value => value + 100);
setPreparedImageScalars(image, { ...prepared, shape: [4, 3, 2], values: changed });
assert.notEqual(image.getPointData().getScalars(), reshaped);
assert.deepEqual([...image.getPointData().getScalars().getData()], [...changed]);

const packed = Float32Array.from({ length: 48 }, (_, index) => index);
setPreparedImageScalars(image, { ...prepared, values: packed.subarray(0, 24) });
const firstSpan = image.getPointData().getScalars();
setPreparedImageScalars(image, { ...prepared, values: packed.subarray(24) });
assert.notEqual(image.getPointData().getScalars(), firstSpan);
assert.deepEqual([...image.getPointData().getScalars().getData()], [...packed.subarray(24)]);

const vectorImage = vtkImageData.newInstance();
const vectorPrepared = { ...prepared, values: Float32Array.from({ length: 72 }, (_, index) => index * 0.25) };
setPreparedImageVectors(vectorImage, vectorPrepared);
const vectors = vectorImage.getPointData().getVectors();
const vectorTime = vectors.getMTime();
setPreparedImageVectors(vectorImage, { ...vectorPrepared, origin: [1, 2, 3] });
assert.equal(vectorImage.getPointData().getVectors(), vectors);
assert.equal(vectors.getMTime(), vectorTime);
assert.equal(vectors.getNumberOfComponents(), 3);
assert.deepEqual([...vectors.getData()], [...vectorPrepared.values]);
setPreparedImageVectors(vectorImage, { ...vectorPrepared, values: vectorPrepared.values.slice() });
assert.notEqual(vectorImage.getPointData().getVectors(), vectors);
image.delete();
vectorImage.delete();
console.log("Prepared scalar/vector identity, exact spans, shape replacement, mask changes, geometry-only updates, and invalid-input atomicity passed");
