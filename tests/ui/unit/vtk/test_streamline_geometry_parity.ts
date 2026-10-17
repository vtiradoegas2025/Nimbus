import {
  streamlineGeometryFingerprint,
  streamlineSeedFingerprint,
} from "../../../../ui/src/lib/vtk/streamlineGeometryParity.ts";
import type { StreamlineExtractionResult } from "../../../../ui/src/lib/vtk/streamlineExtractionProtocol.ts";

const result = (offset = 0): StreamlineExtractionResult => ({
  forward: {
    path: {
      points: new Float32Array([0, 0, 0, 1 + offset, 0, 0]),
      lines: new Uint32Array([2, 0, 1]),
      polys: new Uint32Array(),
      strips: new Uint32Array(),
      normals: new Float32Array(),
      scalars: new Float32Array([1, 1]),
    },
    arrows: null,
  },
  backward: { path: null, arrows: null },
  representation: "tubes",
  integratedPoints: 2,
  boundaryContacts: { lower: [false, false, false], upper: [false, false, false] },
  knownScratchBytes: 0,
  elapsedMs: 1,
});

const baseline = streamlineGeometryFingerprint(result());
if (baseline !== streamlineGeometryFingerprint(result())) {
  throw new Error("Equivalent streamline geometry does not have a stable fingerprint");
}
if (baseline === streamlineGeometryFingerprint(result(0.25))) {
  throw new Error("Streamline geometry drift is not detected");
}
if (streamlineGeometryFingerprint(null) !== null) {
  throw new Error("Missing streamline geometry must not advertise a fingerprint");
}
const stripChanged = result();
stripChanged.forward.path!.strips = new Uint32Array([3, 0, 1, 0]);
if (baseline === streamlineGeometryFingerprint(stripChanged)) {
  throw new Error("Streamline strip topology drift is not detected");
}
if (streamlineSeedFingerprint(new Float32Array([1, 2, 3]))
    === streamlineSeedFingerprint(new Float32Array([1, 2, 4]))) {
  throw new Error("Streamline seed drift is not detected");
}
const normalChanged = result();
normalChanged.forward.path!.normals = new Float32Array([0, 0, 1, 0, 0, 1]);
if (baseline === streamlineGeometryFingerprint(normalChanged)) throw new Error("Streamline normal drift is not detected");
console.log("\nstreamline geometry parity tests: 6 passed, 0 failed.");
