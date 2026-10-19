import assert from "node:assert/strict";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData.js";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray.js";
import vtkImageMarchingCubes from "@kitware/vtk.js/Filters/General/ImageMarchingCubes.js";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData.js";
import { finiteImageMarchingCubes } from "../../../../ui/src/lib/vtk/finiteImageMarchingCubes.ts";
import { finiteImageStreamline } from "../../../../ui/src/lib/vtk/finiteImageStreamline.ts";
import { finiteVectorSupport } from "../../../../ui/src/lib/vtk/finiteVectorSupport.ts";
import { prepareVtkRequest, dominantVortexCenterline } from "../../../../ui/src/lib/vtk/vtkPreparation.ts";
import { vtkPreparationIdentity } from "../../../../ui/src/lib/vtk/vtkPreparationClient.ts";
import { handoffVtkExtractionDataset } from "../../../../ui/src/lib/vtk/vtkDatasetHandoff.ts";

const shape: [number, number, number] = [5, 7, 4];
const grid = { drM: 1, dyM: 1, dzM: 1, scalarOriginM: [0, 0, 0] as [number, number, number] };
const field = (missing: boolean, uniform = false) => ({ shape, data: Float32Array.from({ length: 140 }, (_, index) => {
  const y = Math.floor(index / 4) % 7, x = Math.floor(index / 28);
  return missing && y === 3 ? NaN : uniform ? 5 : x;
}) });
const missing = prepareVtkRequest({ kind: "scalar", field: field(true), grid, includeValidity: false });
assert.equal(missing.validity, undefined);
assert.equal(missing.values.filter(value => !Number.isFinite(value)).length, 20);
const image = vtkImageData.newInstance(); image.setDimensions(...shape);
const scalars = vtkDataArray.newInstance({ values: missing.values, numberOfComponents: 1 });
image.getPointData().setScalars(scalars);
const marching = finiteImageMarchingCubes({ contourValue: 1.5, computeNormals: true, mergePoints: true });
assert.ok(Object.isFrozen(marching));
marching.setInputData(image);
const output = marching.getOutputData();
const points = output.getPoints().getData() as Float32Array, triangles = output.getPolys().getData() as Uint32Array;
const normals = output.getPointData().getNormals().getData() as Float32Array;
assert.ok(points.length > 0);
let area = 0;
for (let i = 0; i < points.length; i += 3) {
  assert.equal(points[i], 1.5);
  assert.ok(points[i + 1] <= 2 || points[i + 1] >= 4);
  assert.ok(Math.abs(normals[i] + 1) < 1e-6);
  assert.equal(normals[i + 1], 0); assert.equal(normals[i + 2], 0);
}
for (let i = 0; i < triangles.length; i += 4) {
  const [a,b,c] = [triangles[i+1],triangles[i+2],triangles[i+3]].map(v => v*3);
  area += Math.abs((points[b+1]-points[a+1])*(points[c+2]-points[a+2])-(points[b+2]-points[a+2])*(points[c+1]-points[a+1]))/2;
}
assert.equal(area, 12);
const uniform = prepareVtkRequest({ kind: "scalar", field: field(true, true), grid, includeValidity: false });
scalars.setData(uniform.values); image.modified(); marching.setContourValue(1);
assert.equal(marching.getOutputData().getNumberOfPolys(), 0);
const clean = prepareVtkRequest({ kind: "scalar", field: field(false), grid, includeValidity: false });
scalars.setData(clean.values); image.modified(); marching.setContourValue(1.5);
const reference = vtkImageMarchingCubes.newInstance({ contourValue: 1.5, computeNormals: true, mergePoints: true }); reference.setInputData(image);
assert.deepEqual(marching.getOutputData().getPoints().getData(), reference.getOutputData().getPoints().getData());
assert.deepEqual(marching.getOutputData().getPointData().getNormals().getData(), reference.getOutputData().getPointData().getNormals().getData());
clean.values.fill(NaN); scalars.modified(); image.modified();
assert.equal(marching.getOutputData().getNumberOfPolys(), 0);
const gpu = prepareVtkRequest({ kind: "scalar", field: field(true), grid });
const retained = gpu.values.slice();
const handoff = handoffVtkExtractionDataset(gpu.values, "masked-contour", gpu.validity);
assert.deepEqual(handoff.values, missing.values); assert.deepEqual(gpu.values, retained);
assert.equal(handoff.report.copiedBytes, gpu.values.byteLength);
const identityField = field(false);
assert.notEqual(vtkPreparationIdentity({ kind: "scalar", field: identityField, grid, includeValidity: false }), vtkPreparationIdentity({ kind: "scalar", field: identityField, grid }));

const vectorShape: [number, number, number] = [10, 3, 3];
const component = (value: number, gap: boolean) => ({ shape: vectorShape, data: Float32Array.from({ length: 90 }, (_, i) => gap && Math.floor(i/9) === 3 ? NaN : value) });
const vectors = prepareVtkRequest({ kind: "vector", fields: [component(10, false), component(0, true), component(0, false)], grid });
assert.equal(vectors.values.filter(value => !Number.isFinite(value)).length, 9);
const support = finiteVectorSupport(vectors.values, vectorShape, [0,0,0], [1,1,1]);
const velocity = new Float32Array([1,2,3]);
assert.equal(support.sample([3,1,1], velocity), false); assert.deepEqual([...velocity], [1,2,3]);
assert.ok(support.sample([2,1,1], velocity)); assert.deepEqual([...velocity], [10,0,0]);
assert.ok(!support.connects([1,1,1], [8,1,1])); assert.ok(!support.connects([8,1,1], [1,1,1]));
assert.ok(support.sample([4.5,1,1], velocity));
assert.ok(support.connects([5,1,1], [8,1,1])); assert.ok(support.connects([5,1,1], [5,1,1]));
const vectorImage = vtkImageData.newInstance(); vectorImage.setDimensions(...vectorShape);
const vectorArray = vtkDataArray.newInstance({ values: vectors.values, numberOfComponents: 3 }); vectorImage.getPointData().setVectors(vectorArray);
const integration = finiteImageStreamline({ integrationStep: 0.7, maximumNumberOfSteps: 20 }, vectorImage);
assert.ok(Object.isFrozen(integration));
const position = [1,1,1]; assert.equal(integration.computeNextStep(vectorArray, vectorImage, 0.7, position), false); assert.deepEqual(position, [1,1,1]);
const seeds = vtkPolyData.newInstance(); seeds.getPoints().setData(new Float32Array([1,1,1,5,1,1]), 3);
const slow = finiteImageStreamline({ integrationStep: 0.05, maximumNumberOfSteps: 30 }, vectorImage);
slow.setInputData(vectorImage); slow.setInputData(seeds, 1);
const paths = slow.getOutputData(); const linePoints = paths.getPoints().getData();
assert.equal(paths.getLines().getNumberOfCells(), 2);
assert.ok(linePoints.length > 6);
for (let i = 0; i < linePoints.length; i += 3) assert.ok(linePoints[i] <= 2 || linePoints[i] >= 5);
for (const object of [marching,reference,image,scalars,vectorImage,vectorArray,integration,slow,seeds]) object.delete();
const ridge = { shape: [3,3,7] as [number,number,number], spacing: [100,100,100] as [number,number,number],
  origin: [0,0,0] as [number,number,number], data: Float32Array.from({ length: 63 }, (_, i) => i % 7 === 3 ? NaN : 10 - i % 7) };
const centerline = dominantVortexCenterline(ridge, "q-criterion");
assert.equal(centerline.length, 9);
assert.ok(centerline.filter((_,i) => i%3 === 2).every(z => z <= 200));
const anisotropicGrid = { drM: 2, dyM: 3, dzM: 4, scalarOriginM: [10,20,30] as [number,number,number] };
const affine = { shape, data: Float32Array.from({length:140}, (_, i) => {
  const x = Math.floor(i/28), y = Math.floor(i/4)%7, z = i%4;
  return y === 3 ? NaN : 2*(10+2*x)+3*(20+3*y)+5*(30+4*z);
}) };
const physical = prepareVtkRequest({ kind:"scalar", field:affine, grid:anisotropicGrid, includeValidity:false });
const physicalImage = vtkImageData.newInstance(); physicalImage.setDimensions(...shape); physicalImage.setSpacing(physical.spacing); physicalImage.setOrigin(physical.origin);
physicalImage.getPointData().setScalars(vtkDataArray.newInstance({values:physical.values,numberOfComponents:1}));
const physicalContour = finiteImageMarchingCubes({contourValue:276,computeNormals:true,mergePoints:true}); physicalContour.setInputData(physicalImage);
const physicalOutput = physicalContour.getOutputData(), physicalPoints = physicalOutput.getPoints().getData(), physicalNormals = physicalOutput.getPointData().getNormals().getData();
assert.ok(physicalPoints.length > 0);
for(let i=0;i<physicalPoints.length;i+=3) {
  assert.ok(Math.abs(2*physicalPoints[i]+3*physicalPoints[i+1]+5*physicalPoints[i+2]-276)<1e-4);
  for(let axis=0;axis<3;axis++) assert.ok(Math.abs(physicalNormals[i+axis]+[2,3,5][axis]/Math.sqrt(38))<1e-6);
}
physicalContour.delete(); physicalImage.delete();
console.log("Finite contour area/normals, no false caps, clean parity, array replacement, missing-vector preservation, and gap-safe integration passed");
