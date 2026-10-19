import {
  scalarMultisetFingerprint,
  scalarQuantizedMultisetFingerprint,
} from "../../../../ui/src/lib/vtk/scalarCompositionParity.ts";

const full = new Float32Array([1, -2, 3, 4, 5]);
const bricks = [new Float32Array([4, 1]), new Float32Array([5, 3, -2])];
if (scalarMultisetFingerprint([full]) !== scalarMultisetFingerprint(bricks)) {
  throw new Error("Equivalent scalar brick composition does not match the monolithic fingerprint");
}
if (scalarMultisetFingerprint([full]) === scalarMultisetFingerprint([
  new Float32Array([1, -2, 3, 4, 6]),
])) {
  throw new Error("Scalar composition drift is not detected");
}
if (scalarQuantizedMultisetFingerprint([new Float32Array([1, 2])])
    !== scalarQuantizedMultisetFingerprint([new Float32Array([1 + 1e-7, 2 - 1e-7])])) {
  throw new Error("Sub-quantum scalar drift should remain numerically equivalent");
}
console.log("\nscalar composition parity tests: 3 passed, 0 failed.");
