export interface JointVolumeTopology {
  roles: readonly number[];
  temporalPeers: readonly number[];
  temporalMasked: readonly number[];
}

const roleDeclaration = "uniform vec4 nimbusReferences, nimbusRoles, nimbusReach, nimbusShadowSteps, nimbusSampleSteps;";
const temporalDeclaration = "uniform vec4 nimbusTemporalPeers, nimbusScalarScales, nimbusScalarOffsets, nimbusTemporalMasked;";

export function jointVolumeTopologyKey(topology: JointVolumeTopology): string {
  for (const [values, minimum, maximum] of [
    [topology.roles, -1, 4], [topology.temporalPeers, -1, 3], [topology.temporalMasked, 0, 1],
  ] as const) {
    if (values.length !== 4 || values.some(value => !Number.isSafeInteger(value) || value < minimum || value > maximum)) {
      throw new Error("Invalid joint volume role, peer, or validity topology");
    }
  }
  return JSON.stringify([topology.roles, topology.temporalPeers, topology.temporalMasked]);
}

function omitBlock(source: string, prefix: string): string {
  const start = source.indexOf(prefix);
  if (start < 0) throw new Error("Joint volume history shader branch is unavailable");
  const brace = source.indexOf("{", start);
  let depth = 1;
  for (let end = brace + 1; end < source.length; end++) {
    if (source[end] === "{") depth++;
    if (source[end] === "}" && --depth === 0) return source.slice(0, start) + source.slice(end + 1);
  }
  throw new Error("Joint volume history shader branch is incomplete");
}

export function specializeJointVolumeTopology(source: string, topology: JointVolumeTopology): string {
  jointVolumeTopologyKey(topology);
  if (!source.includes(roleDeclaration) || !source.includes(temporalDeclaration)) {
    throw new Error("Joint volume topology declarations are unavailable");
  }
  for (let index = 0; index < 4; index++) if (topology.roles[index] === 4) {
    source = omitBlock(source, `if (nimbusRoles[${index}] != 4.0 && nimbusSampleValid(pos, ${index}))`);
    source = omitBlock(source, `if (coefficients[${index}] > 0.0)`);
  }
  const vector = (values: readonly number[]) => `vec4(${values.map(value => `${value}.0`).join(", ")})`;
  source = source.replace(roleDeclaration,
    `uniform vec4 nimbusReferences, nimbusReach, nimbusShadowSteps, nimbusSampleSteps;\nconst vec4 nimbusRoles = ${vector(topology.roles)};`);
  source = source.replace(temporalDeclaration,
    `uniform vec4 nimbusScalarScales, nimbusScalarOffsets;\nconst vec4 nimbusTemporalPeers = ${vector(topology.temporalPeers)};\nconst vec4 nimbusTemporalMasked = ${vector(topology.temporalMasked)};`);
  for (const [name, values] of [["nimbusRoles", topology.roles], ["nimbusTemporalPeers", topology.temporalPeers],
    ["nimbusTemporalMasked", topology.temporalMasked]] as const) {
    source = source.replace(new RegExp(`${name}\\[([0-3])\\]`, "g"), (_, index: string) => `${values[Number(index)]}.0`);
  }
  return source;
}
