/*
 * Tests for src/lib/npy.ts parsing — in particular the zero-copy fast path.
 *
 * Run via:
 *   npm --prefix ui run test:npy
 *
 * The body offset of a v1.0 .npy is padded to a 64-byte boundary, so a field
 * loaded from a fresh buffer can be viewed in place (no 21 MB memcpy per
 * field). This verifies the values are correct AND that the aligned path
 * returns a view over the source buffer, while a misaligned source still
 * parses correctly via the copy fallback.
 */

import { parseNpy, parseNpy3D, fieldPercentiles, slice2D } from "../../../../ui/src/lib/platform/npy.ts";
import { generateNpyValidationCases } from "../../data/npy_validation_cases.mjs";

let passed = 0;
let failed = 0;
const failures: string[] = [];

function assertTrue(cond: boolean, label: string): void {
  if (cond) passed++;
  else {
    failed++;
    failures.push(`FAIL ${label}`);
  }
}

/** Build a valid v1.0 `<f4` C-order .npy as an ArrayBuffer. */
function buildNpy(shape: number[], values: number[]): ArrayBuffer {
  const shapeStr =
    shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  return buildRawNpy(shapeStr, values.length, 1, values);
}

function buildRawNpy(
  shapeText: string,
  bodyElements: number,
  major: number,
  values: number[] = [],
): ArrayBuffer {
  const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shapeText}), }`;
  const prefixBytes = major === 2 ? 12 : 10;
  // Pad with spaces so the body begins on a 64-byte boundary.
  const pad = (64 - ((prefixBytes + dict.length + 1) % 64)) % 64;
  const headerStr = dict + " ".repeat(pad) + "\n";
  const headerBytes = new TextEncoder().encode(headerStr);
  const headerLen = headerBytes.length;

  const total = prefixBytes + headerLen + bodyElements * 4;
  const buf = new ArrayBuffer(total);
  const u8 = new Uint8Array(buf);
  u8.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0); // \x93NUMPY
  u8[6] = major;
  u8[7] = 0; // minor
  u8[8] = headerLen & 0xff;
  u8[9] = (headerLen >> 8) & 0xff;
  if (major === 2) {
    u8[10] = (headerLen >> 16) & 0xff;
    u8[11] = (headerLen >> 24) & 0xff;
  }
  u8.set(headerBytes, prefixBytes);
  const dv = new DataView(buf, prefixBytes + headerLen);
  for (let i = 0; i < values.length; i++) dv.setFloat32(i * 4, values[i], true);
  return buf;
}

// ---------- correct values + shape ----------
{
  const buf = buildNpy([2, 3], [1, 2, 3, 4, 5, 6]);
  const arr = parseNpy(buf);
  assertTrue(arr.shape.join(",") === "2,3", "shape parsed");
  assertTrue(arr.data.length === 6, "element count");
  assertTrue(
    [...arr.data].join(",") === "1,2,3,4,5,6",
    "values round-trip",
  );
  assertTrue(
    arr.descriptor?.decodedBytes === 24 && arr.descriptor.rank === 2,
    "validated descriptor reports rank and decoded bytes",
  );
  assertTrue(
    parseNpy(buf, { expectedShapes: [[2, 3]] }).data.length === 6,
    "manifest shape agreement accepts an exact shape",
  );
  let mismatchThrew = false;
  try {
    parseNpy(buf, { expectedShapes: [[2, 3, 4], [2, 4]] });
  } catch {
    mismatchThrew = true;
  }
  assertTrue(mismatchThrew, "manifest shape disagreement is rejected");
}

// ---------- corpus shared with the Rust parser ----------
{
  const cases = generateNpyValidationCases();
  assertTrue(cases.length === 20, "shared validation corpus retains all 20 cases");
  for (const testCase of cases) {
    const input = testCase.bytes.buffer;
    let accepted = true;
    try {
      parseNpy(input);
    } catch {
      accepted = false;
    }
    assertTrue(
      accepted === testCase.accepted,
      `shared validation corpus: ${testCase.name}`,
    );
  }
}

// ---------- aligned source -> zero-copy view over the same buffer ----------
{
  const buf = buildNpy([4], [10, 20, 30, 40]);
  const arr = parseNpy(buf);
  assertTrue(arr.data.buffer === buf, "aligned parse is a zero-copy view");
  assertTrue(arr.data.byteOffset % 4 === 0, "view offset is 4-byte aligned");
  assertTrue([...arr.data].join(",") === "10,20,30,40", "view values correct");
}

// ---------- misaligned source -> copy fallback, still correct ----------
{
  const inner = buildNpy([3], [7, 8, 9]);
  // Place the npy at byteOffset 1 of a larger buffer so the body offset is odd.
  const outer = new ArrayBuffer(inner.byteLength + 1);
  new Uint8Array(outer).set(new Uint8Array(inner), 1);
  const view = new Uint8Array(outer, 1, inner.byteLength);
  const arr = parseNpy(view);
  assertTrue([...arr.data].join(",") === "7,8,9", "misaligned values correct");
  assertTrue(arr.data.buffer !== outer, "misaligned parse copies to a fresh buffer");
}

// ---------- parseNpy3D rank guard ----------
{
  const buf = buildNpy([2, 2], [1, 2, 3, 4]);
  let threw = false;
  try {
    parseNpy3D(buf);
  } catch {
    threw = true;
  }
  assertTrue(threw, "parseNpy3D rejects a 2D field");
}

// ---------- slice display orientation ----------
{
  const field = {
    shape: [2, 3, 4] as [number, number, number],
    data: Float32Array.from({ length: 24 }, (_, index) => index),
  };
  const ySection = slice2D(field, 1, 1);
  assertTrue(ySection.rowAxis === 2 && ySection.colAxis === 0,
    "y slice maps height to screen y and x to screen x");
  assertTrue(ySection.rows === 4 && ySection.cols === 2,
    "y slice output is height rows by horizontal columns");
  assertTrue(ySection.data.map((row) => row.join(",")).join("|") === "4,16|5,17|6,18|7,19",
    "y slice transposes C-order values without changing samples");
  const xSection = slice2D(field, 0, 1);
  assertTrue(xSection.rowAxis === 2 && xSection.colAxis === 1,
    "x slice maps height to screen y and y to screen x");
  assertTrue(xSection.data.map((row) => row.join(",")).join("|") === "12,16,20|13,17,21|14,18,22|15,19,23",
    "x slice transposes C-order values without changing samples");
}

// ---------- fieldPercentiles (robust range for volume transfer fns) ----------
{
  // 0..99: p50 ~ midpoint, p2 near the bottom, p98 near the top.
  const data = new Float32Array(100);
  for (let i = 0; i < 100; i++) data[i] = i;
  const field = { shape: [100], data };
  const [p2, p50, p98] = fieldPercentiles(field, [0.02, 0.5, 0.98]);
  assertTrue(p2 <= 5 && p2 >= 0, "p2 near the low end");
  assertTrue(p50 >= 45 && p50 <= 54, "p50 near the middle");
  assertTrue(p98 >= 95 && p98 <= 99, "p98 near the high end");
  assertTrue(p2 < p50 && p50 < p98, "percentiles are monotone");
}
{
  // Heavy tail: 100 zeros + one outlier. A [min,max] range would be [0,1000];
  // the robust p98 must clip the outlier to keep structure visible.
  const data = new Float32Array(101);
  data[100] = 1000;
  const field = { shape: [101], data };
  const [p98] = fieldPercentiles(field, [0.98]);
  assertTrue(p98 === 0, "p98 clips a lone outlier (robust range)");
}
{
  // All non-finite -> NaN percentiles (caller falls back to absolute range).
  const data = new Float32Array([NaN, NaN, Number.POSITIVE_INFINITY]);
  const field = { shape: [3], data };
  const [p50] = fieldPercentiles(field, [0.5]);
  assertTrue(Number.isNaN(p50), "all-non-finite -> NaN");
}

console.log(`\nnpy tests: ${passed} passed, ${failed} failed.`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log("  " + f);
  process.exit(1);
}
process.exit(0);
