import { readFileSync } from "node:fs";
import {
  assertVtkArrayAllocation,
  releaseVtkPointDataArrays,
  replaceVtkPointDataArray,
  VTK_SCALAR_SLOT,
  VTK_VECTOR_SLOT,
  type OwnedVtkArray,
  type VtkPointDataOwner,
} from "../../../../ui/src/lib/vtk/vtkArrayOwnership.ts";
import { MAX_FIELD_DECODED_BYTES } from "../../../../ui/src/lib/platform/npy.ts";

let passed = 0;
let failed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

class MockArray implements OwnedVtkArray {
  deletes = 0;
  private readonly name: string;
  constructor(name: string) { this.name = name; }
  getName(): string { return this.name; }
  delete(): void { this.deletes++; }
}

class MockPointData implements VtkPointDataOwner {
  arrays: OwnedVtkArray[] = [];
  active: "scalars" | "vectors" | null = null;
  failNextAttach = false;
  getArrays(): OwnedVtkArray[] { return this.arrays; }
  removeAllArrays(): void { this.arrays = []; this.active = null; }
  setScalars(array: OwnedVtkArray): boolean { this.add(array); this.active = "scalars"; return true; }
  setVectors(array: OwnedVtkArray): boolean { this.add(array); this.active = "vectors"; return true; }
  private add(array: OwnedVtkArray): void {
    if (this.failNextAttach) {
      this.failNextAttach = false;
      throw new Error("attach failed");
    }
    const index = this.arrays.findIndex((current) => current.getName() === array.getName());
    if (index >= 0) this.arrays[index] = array;
    else this.arrays.push(array);
  }
}

{
  const pointData = new MockPointData();
  const first = new MockArray(VTK_SCALAR_SLOT);
  const second = new MockArray(VTK_SCALAR_SLOT);
  replaceVtkPointDataArray(pointData, "scalars", first);
  replaceVtkPointDataArray(pointData, "scalars", second);
  check(pointData.arrays.length === 1, "repeated scalar replacement keeps exactly one array");
  check(pointData.arrays[0] === second, "the newest scalar owns the stable slot");
  check(first.deletes === 1, "the displaced same-name scalar is explicitly deleted");
  check(second.deletes === 0, "the attached scalar remains live");
}

{
  const pointData = new MockPointData();
  const generations: MockArray[] = [];
  for (let i = 0; i < 100; i++) {
    const array = new MockArray(VTK_SCALAR_SLOT);
    generations.push(array);
    replaceVtkPointDataArray(pointData, "scalars", array);
  }
  check(pointData.arrays.length === 1, "100 field switches plateau at one resident VTK array");
  check(
    generations.slice(0, -1).every((array) => array.deletes === 1),
    "every displaced array in the switch soak is released",
  );
  check(generations.at(-1)?.deletes === 0, "only the active soak array remains owned");
}

{
  const pointData = new MockPointData();
  const legacyA = new MockArray("temperature");
  const legacyB = new MockArray("vorticity");
  pointData.arrays = [legacyA, legacyB];
  const next = new MockArray(VTK_VECTOR_SLOT);
  replaceVtkPointDataArray(pointData, "vectors", next);
  check(pointData.arrays.length === 1 && pointData.arrays[0] === next, "legacy field-name arrays are collapsed to one slot");
  check(legacyA.deletes === 1 && legacyB.deletes === 1, "every legacy array is deleted once");
  check(pointData.active === "vectors", "vector replacement restores the active vector attribute");
}

{
  const pointData = new MockPointData();
  const current = new MockArray(VTK_SCALAR_SLOT);
  pointData.arrays = [current];
  const rejected = new MockArray(VTK_SCALAR_SLOT);
  pointData.failNextAttach = true;
  let threw = false;
  try { replaceVtkPointDataArray(pointData, "scalars", rejected); } catch { threw = true; }
  check(threw, "an attach failure remains visible to the caller");
  check(pointData.arrays[0] === current && current.deletes === 0, "attach failure preserves the current array");
  check(rejected.deletes === 1, "attach failure releases the unattached replacement");
}

{
  const pointData = new MockPointData();
  const a = new MockArray("a");
  const b = new MockArray("b");
  pointData.arrays = [a, b, a];
  releaseVtkPointDataArrays(pointData);
  check(pointData.arrays.length === 0, "data-set release detaches every point-data array");
  check(a.deletes === 1 && b.deletes === 1, "data-set release deletes duplicate references once");
}

{
  assertVtkArrayAllocation(MAX_FIELD_DECODED_BYTES / 4, 1);
  let oversize = false;
  try { assertVtkArrayAllocation(MAX_FIELD_DECODED_BYTES / 4 + 1, 1); } catch { oversize = true; }
  check(oversize, "an oversize derived scalar fails before allocation");
  let invalid = false;
  try { assertVtkArrayAllocation(Number.MAX_SAFE_INTEGER, 3); } catch { invalid = true; }
  check(invalid, "unsafe derived-array arithmetic is rejected");
}

{
  const plotFiles = [
    "VolumeRender3D.tsx",
    "Isosurface3D.tsx",
    "Streamlines3D.tsx",
    "Trajectories3D.tsx",
    "SupercellRender3D.tsx",
  ];
  const sources = plotFiles.map((file) =>
    readFileSync(new URL(`../../../../ui/src/components/plots/${file}`, import.meta.url), "utf8")
  );
  check(
    sources.every((source) => !source.includes(".getPointData().setScalars(")),
    "plot components have no direct scalar append path",
  );
  check(
    sources.every((source) => !source.includes(".getPointData().setVectors(")),
    "plot components have no direct vector append path",
  );
  const imageHelper = readFileSync(new URL("../../../../ui/src/lib/vtk/vtkImageData.ts", import.meta.url), "utf8");
  check(imageHelper.includes("name: VTK_SCALAR_SLOT"), "image scalars use the stable internal slot");
  check(imageHelper.includes("name: VTK_VECTOR_SLOT"), "image vectors use the stable internal slot");
  check(imageHelper.includes("assertVtkArrayAllocation"), "VTK packing is guarded by the decoded-byte ceiling");
}

console.log(`\nvtkArrayOwnership tests: ${passed} passed, ${failed} failed.`);
if (failures.length) {
  console.log(failures.join("\n"));
  process.exit(1);
}
