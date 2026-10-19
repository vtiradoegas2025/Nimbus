export type SceneFieldRole = "detail" | "context" | "auxiliary";

export interface SceneFieldBudgetInput {
  identity: string;
  role: SceneFieldRole;
  bytesPerCell: number;
}

export interface SupercellSceneBudget {
  detailCells: number;
  contextCells: number;
  auxiliaryCells: number;
  estimatedPeakBytes: number;
  planningEnvelopeBytes: number;
  uniqueFields: number;
}

const ROLES: readonly SceneFieldRole[] = ["detail", "context", "auxiliary"];
const MIB = 1024 ** 2;

export function planSupercellSceneBudget(input: {
  baseCells: number;
  paneScale: number;
  systemRamBytes: number | null;
  qualifiedCapacityBytes?: number | null;
  fields: readonly SceneFieldBudgetInput[];
  preparedChannels: Partial<Record<SceneFieldRole, number>>;
  scratchChannels?: Partial<Record<SceneFieldRole, number>>;
}): SupercellSceneBudget {
  if (!Number.isSafeInteger(input.baseCells) || input.baseCells < 8
      || !Number.isFinite(input.paneScale) || input.paneScale <= 0 || input.paneScale > 1) {
    throw new Error("Invalid scene cell or pane budget");
  }
  const fields = new Map<string, SceneFieldBudgetInput>();
  for (const field of input.fields) {
    if (!field.identity || !ROLES.includes(field.role)
        || !Number.isSafeInteger(field.bytesPerCell) || field.bytesPerCell < 4) {
      throw new Error("Invalid scene field budget");
    }
    const key = JSON.stringify([field.role, field.identity]);
    const previous = fields.get(key);
    if (previous && previous.bytesPerCell !== field.bytesPerCell) {
      throw new Error("Shared scene field has conflicting storage costs");
    }
    fields.set(key, field);
  }
  const cost = { detail: 0, context: 0, auxiliary: 0 };
  // A committed generation stays visible while its replacement and worker input are staged.
  for (const field of fields.values()) cost[field.role] += field.bytesPerCell * 3;
  for (const role of ROLES) {
    const prepared = input.preparedChannels[role] ?? 0;
    const scratch = input.scratchChannels?.[role] ?? 0;
    if (!Number.isSafeInteger(prepared) || prepared < 0
        || !Number.isSafeInteger(scratch) || scratch < 0) {
      throw new Error("Invalid scene preparation budget");
    }
    cost[role] += prepared * 20 + scratch * 4;
  }
  const knownRam = input.systemRamBytes != null && Number.isSafeInteger(input.systemRamBytes)
    && input.systemRamBytes > 0;
  const viewerEnvelope = knownRam
    ? Math.min(4096 * MIB, Math.max(256 * MIB, Math.floor(input.systemRamBytes! / 8)))
    : 512 * MIB;
  const qualified = input.qualifiedCapacityBytes;
  if (qualified != null && (!Number.isSafeInteger(qualified) || qualified < 1)) {
    throw new Error("Invalid qualified scene capacity");
  }
  const planningEnvelopeBytes = Math.floor(Math.min(viewerEnvelope, qualified ?? Infinity) * input.paneScale);
  const contextCells = cost.context > 0 ? Math.max(8, Math.floor(input.baseCells * 0.25)) : 8;
  const desired = {
    detail: cost.context > 0 ? Math.max(8, input.baseCells - contextCells) : input.baseCells,
    context: contextCells,
    auxiliary: input.baseCells,
  };
  const estimate = (cells: Record<SceneFieldRole, number>) => ROLES.reduce(
    (sum, role) => sum + cells[role] * cost[role], 0,
  );
  const minimum = estimate({ detail: 8, context: 8, auxiliary: 8 });
  if (!Number.isSafeInteger(minimum) || minimum > planningEnvelopeBytes) {
    throw new Error("Scene working set exceeds the planning envelope");
  }
  const desiredBytes = estimate(desired);
  if (!Number.isSafeInteger(desiredBytes)) throw new Error("Scene working set exceeds integer precision");
  const fraction = desiredBytes > planningEnvelopeBytes
    ? (planningEnvelopeBytes - minimum) / (desiredBytes - minimum) : 1;
  const cells = Object.fromEntries(ROLES.map(role => [role,
    8 + Math.floor((desired[role] - 8) * fraction),
  ])) as Record<SceneFieldRole, number>;
  return {
    detailCells: cells.detail,
    contextCells: cells.context,
    auxiliaryCells: cells.auxiliary,
    estimatedPeakBytes: estimate(cells),
    planningEnvelopeBytes,
    uniqueFields: fields.size,
  };
}
