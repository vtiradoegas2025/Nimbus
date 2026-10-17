import {
  appearanceFingerprint,
  canonicalAppearanceJson,
  type VtkAppearanceValue,
} from "../../../../ui/src/lib/vtk/vtkAppearanceEvidence.ts";

let passed = 0;

function check(condition: boolean, message: string): void {
  if (!condition) throw new Error(message);
  passed += 1;
}

const ordered: VtkAppearanceValue = {
  plot: "volume",
  nested: { z: 3, a: 1 },
  values: [1 / 3, -0, Number.POSITIVE_INFINITY],
};
const reordered: VtkAppearanceValue = {
  values: [0.3333333333334, 0, null],
  nested: { a: 1, z: 3 },
  plot: "volume",
};

check(
  canonicalAppearanceJson(ordered)
    === '{"nested":{"a":1,"z":3},"plot":"volume","values":[0.333333333333,0,null]}',
  "canonical evidence sorts keys and normalizes floating-point noise",
);
check(
  appearanceFingerprint(ordered) === appearanceFingerprint(reordered),
  "equivalent scene state has one deterministic fingerprint",
);
check(
  appearanceFingerprint({ ...reordered, plot: "isosurface" })
    !== appearanceFingerprint(reordered),
  "a semantic render change changes the evidence fingerprint",
);
check(
  /^fnv1a64:[a-f0-9]{16}$/.test(appearanceFingerprint(ordered)),
  "evidence fingerprints use the bounded repository digest form",
);

console.log(`vtk appearance evidence: ${passed} checks passed`);
