import assert from "node:assert/strict";
import { JOINT_VOLUME_DECLARATIONS, JOINT_VOLUME_FUNCTIONS, JOINT_VOLUME_SAMPLERS } from "../../../../ui/src/lib/vtk/jointVolumeShader.ts";
import { jointVolumeTopologyKey, specializeJointVolumeTopology } from "../../../../ui/src/lib/vtk/jointVolumeTopology.ts";

const source = [JOINT_VOLUME_DECLARATIONS, JOINT_VOLUME_SAMPLERS, JOINT_VOLUME_FUNCTIONS].join("\n");
const topology = { roles: [0, 1, 4, 4], temporalPeers: [2, 3, -1, -1], temporalMasked: [1, 0, 1, 0] };
const snapshot = JSON.stringify(topology);
const result = specializeJointVolumeTopology(source, topology);
assert.equal(JSON.stringify(topology), snapshot);
assert.ok(result.includes("const vec4 nimbusTemporalPeers = vec4(2.0, 3.0, -1.0, -1.0)"));
assert.ok(result.includes("const vec4 nimbusRoles = vec4(0.0, 1.0, 4.0, 4.0)"));
assert.ok(result.includes("const vec4 nimbusTemporalMasked = vec4(1.0, 0.0, 1.0, 0.0)"));
assert.ok(result.includes("int peer = int(2.0)") && result.includes("int peer = int(3.0)"));
assert.ok(result.includes("nimbusRawScalar2(pos)") && result.includes("nimbusRawScalar3(pos)"));
assert.ok(result.includes("nimbusPhysicalScalar(pos, peer)"));
assert.ok(result.includes("nimbusRawValid(pos, peer)") && result.includes("nimbusFixedValid(pos, peer)"));
assert.ok(result.includes("nimbusTemporalMasked[peer]"));
for (const index of [2, 3]) {
  assert.ok(!result.includes(`if (coefficients[${index}] > 0.0)`));
  assert.ok(!result.includes(`getColorFromTexture(nimbusScalar${index}(posVC)`));
}
for (const index of [0, 1]) assert.ok(result.includes(`if (coefficients[${index}] > 0.0)`));
assert.ok(result.includes("nimbusScalar0(posVC + offset)") && result.includes("nimbusScalar1(posVC - offset)"));
assert.equal(result.match(/\{/g)?.length, result.match(/\}/g)?.length);

let cases = 0;
for (let code = 0; code < 6 ** 4; code++) {
  let cursor = code;
  const roles = Array.from({ length: 4 }, () => { const role = cursor % 6 - 1; cursor = Math.floor(cursor / 6); return role; });
  const specialized = specializeJointVolumeTopology(source, { ...topology, roles });
  for (let index = 0; index < 4; index++) {
    assert.equal(specialized.includes(`if (coefficients[${index}] > 0.0)`), roles[index] !== 4);
    assert.equal(specialized.includes(`getColorFromTexture(nimbusScalar${index}(posVC)`), roles[index] !== 4);
    assert.ok(specialized.includes(`float nimbusRawScalar${index}(vec3 pos)`));
  }
  assert.equal(specialized.match(/\{/g)?.length, specialized.match(/\}/g)?.length);
  cases++;
}

const key = jointVolumeTopologyKey(topology);
for (const [field, candidates] of [["roles", [-1, 0, 1, 2, 3, 4]], ["temporalPeers", [-1, 0, 1, 2, 3]],
  ["temporalMasked", [0, 1]]] as const) {
  for (let index = 0; index < 4; index++) for (const value of candidates) {
    const changed = { ...topology, [field]: topology[field].map((old, position) => index === position ? value : old) };
    assert.equal(jointVolumeTopologyKey(changed) === key, topology[field][index] === value);
  }
  for (const invalid of [NaN, Infinity, -Infinity, 0.5, -2, 5]) {
    assert.throws(() => jointVolumeTopologyKey({ ...topology, [field]: [invalid, ...topology[field].slice(1)] }), /topology/);
  }
  for (const length of [0, 3, 5]) {
    assert.throws(() => jointVolumeTopologyKey({ ...topology, [field]: Array(length).fill(0) }), /topology/);
  }
}
assert.throws(() => specializeJointVolumeTopology("", topology), /declarations/);
assert.throws(() => specializeJointVolumeTopology(JOINT_VOLUME_DECLARATIONS, topology), /history shader branch/);
console.log(`${cases} role combinations preserve every raw/peer/mask read, remove only history's own coefficient/lighting branches, and rebuild on all topology changes`);
