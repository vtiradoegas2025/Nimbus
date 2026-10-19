import {
  detectStreamlineBoundaryContacts,
  detectSeedBoundaryContacts,
  mergeBrickBoundaryContacts,
} from "../../../../ui/src/lib/vtk/streamlineBoundary.ts";

let passed = 0;
const failures: string[] = [];
function check(condition: boolean, label: string): void {
  if (condition) passed += 1;
  else failures.push(label);
}

const shape: [number, number, number] = [5, 5, 5];
const spacing: [number, number, number] = [2, 3, 4];
const origin: [number, number, number] = [10, 20, 30];
const contacts = detectStreamlineBoundaryContacts(
  new Float32Array([
    14, 26, 38,
    17.5, 26, 38,
    14, 26, 45.5,
  ]),
  new Uint32Array([2, 0, 1, 2, 0, 2]),
  shape,
  spacing,
  origin,
);
check(contacts.upper.join(",") === "true,false,true", "polyline endpoints report reached upper faces");
check(contacts.lower.join(",") === "false,false,false", "seed positions are not mistaken for exits");

const lower = detectStreamlineBoundaryContacts(
  new Float32Array([14, 26, 38, 10.5, 20.5, 31]),
  new Uint32Array([2, 0, 1]),
  shape,
  spacing,
  origin,
);
const merged = mergeBrickBoundaryContacts(contacts, lower);
check(merged.lower.join(",") === "true,true,true", "bidirectional contacts merge every lower face");
check(merged.upper.join(",") === "true,false,true", "bidirectional merge preserves upper faces");

const malformed = detectStreamlineBoundaryContacts(
  new Float32Array([10, 20, 30]),
  new Uint32Array([4, 0]),
  shape,
  spacing,
  origin,
);
check(!malformed.lower.some(Boolean) && !malformed.upper.some(Boolean), "malformed cells fail closed without contacts");
const seedContacts = detectSeedBoundaryContacts(
  new Float32Array([9, 26, 38, 14, 33, 38, 14, 26, 47]),
  shape,
  spacing,
  origin,
);
check(seedContacts.lower.join(",") === "true,false,false",
  "seed envelopes report resident lower-face misses");
check(seedContacts.upper.join(",") === "false,true,true",
  "seed envelopes report resident upper-face misses");

console.log(`\nstreamline boundary tests: ${passed} passed, ${failures.length} failed.`);
if (failures.length > 0) {
  for (const failure of failures) console.error(`FAIL: ${failure}`);
  process.exitCode = 1;
}
