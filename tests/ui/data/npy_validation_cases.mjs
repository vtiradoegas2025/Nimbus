import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const validationSpecs = [
  {
    "name": "valid_2d",
    "kind": "npy",
    "major": 1,
    "shape": "2, 3",
    "bodyElements": 6,
    "accepted": true
  },
  {
    "name": "valid_singleton_tuple",
    "kind": "npy",
    "major": 2,
    "shape": "4,",
    "bodyElements": 4,
    "accepted": true
  },
  {
    "name": "truncated_v1_prefix",
    "kind": "truncated_v1_prefix",
    "accepted": false
  },
  {
    "name": "truncated_v2_prefix",
    "kind": "truncated_v2_prefix",
    "accepted": false
  },
  {
    "name": "max_v1_header_truncated",
    "kind": "max_v1_header_truncated",
    "accepted": false
  },
  {
    "name": "oversized_v2_header",
    "kind": "oversized_v2_header",
    "accepted": false
  },
  {
    "name": "unsupported_major",
    "kind": "npy",
    "major": 3,
    "shape": "2",
    "bodyElements": 2,
    "accepted": false
  },
  {
    "name": "scalar_rank_zero",
    "kind": "npy",
    "major": 1,
    "shape": "",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "zero_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "0, 2",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "negative_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "-1, 2",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "signed_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "+2, 3",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "decimal_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "2.0, 3",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "u64_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "18446744073709551615, 1",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "junk_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "2junk, 3",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "empty_dimension",
    "kind": "npy",
    "major": 1,
    "shape": "2,, 3",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "rank_four",
    "kind": "npy",
    "major": 1,
    "shape": "2, 2, 2, 2",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "dimension_over_limit",
    "kind": "npy",
    "major": 1,
    "shape": "16385",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "product_over_limit",
    "kind": "npy",
    "major": 1,
    "shape": "16384, 16384, 1",
    "bodyElements": 0,
    "accepted": false
  },
  {
    "name": "truncated_body",
    "kind": "npy",
    "major": 1,
    "shape": "2, 3",
    "bodyElements": 5,
    "accepted": false
  },
  {
    "name": "trailing_body",
    "kind": "npy",
    "major": 1,
    "shape": "2, 3",
    "bodyElements": 7,
    "accepted": false
  }
];

const magic = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];

function buildCase(spec) {
  const prefixes = {
    truncated_v1_prefix: [1, 0, 0],
    truncated_v2_prefix: [2, 0, 0, 0],
    max_v1_header_truncated: [1, 0, 0xff, 0xff],
    oversized_v2_header: [2, 0, 1, 0, 1, 0],
  };
  if (spec.kind !== "npy") {
    if (!(spec.kind in prefixes)) throw new Error(`Unknown validation case kind: ${spec.kind}`);
    return new Uint8Array([...magic, ...prefixes[spec.kind]]);
  }
  const major = spec.major ?? 1;
  const prefixBytes = major === 2 ? 12 : 10;
  const dictionary = `{'descr': '<f4', 'fortran_order': False, 'shape': (${spec.shape ?? ""}), }`;
  const padding = (64 - ((prefixBytes + dictionary.length + 1) % 64)) % 64;
  const header = new TextEncoder().encode(dictionary + " ".repeat(padding) + "\n");
  const bytes = new Uint8Array(prefixBytes + header.length + (spec.bodyElements ?? 0) * 4);
  bytes.set([...magic, major, 0]);
  const view = new DataView(bytes.buffer);
  if (major === 2) view.setUint32(8, header.length, true);
  else view.setUint16(8, header.length, true);
  bytes.set(header, prefixBytes);
  return bytes;
}

export function generateNpyValidationCases() {
  return validationSpecs.map(spec => ({ name: spec.name, accepted: spec.accepted, bytes: buildCase(spec) }));
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  process.stdout.write(JSON.stringify(generateNpyValidationCases().map(value => ({
    ...value, bytes: Array.from(value.bytes),
  }))) + "\n");
}
