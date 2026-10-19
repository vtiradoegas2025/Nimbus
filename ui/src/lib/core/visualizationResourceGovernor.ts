/** Workspace-wide ownership and capacity ledger shared by I/O, VTK, and Studio. */
export const VISUALIZATION_RESOURCE_SCHEMA = "nimbus.studio.visualization-resources" as const;
export const VISUALIZATION_RESOURCE_VERSION = 3 as const;

export const VISUALIZATION_RESOURCE_STAGES = [
  "native-read",
  "decoded-field",
  "prepared-main",
  "worker-resident",
  "worker-scratch",
  "handoff",
  "retained-geometry",
  "gpu",
] as const;

export type VisualizationResourceStage = typeof VISUALIZATION_RESOURCE_STAGES[number];
export type VisualizationResourcePriority = "interactive" | "visible" | "playback" | "prefetch" | "background";
export type VisualizationResourceAccounting = "resident" | "reserved";

export interface VisualizationPeakBytes {
  nativeRead: number;
  decoded: number;
  preparedMain: number;
  workerResident: number;
  workerScratch: number;
  handoff: number;
  retainedGeometry: number;
  gpu: number;
}

export interface VisualizationResourcePlan {
  identity: string;
  execution: "worker-cpu" | "native-cpu" | "webgpu-compute";
  representation: "full-domain-lod" | "native-roi" | "hybrid";
  sampledShape: readonly [number, number, number] | null;
  sampledCellUpperBound: number;
  expectedPeakBytes: VisualizationPeakBytes;
  outputBudget: Readonly<{ bytes: number; primitives: number }>;
  provenance: string;
}

export interface VisualizationResourceClaim {
  allocationId: string;
  stage: VisualizationResourceStage;
  bytes: number;
  accounting?: VisualizationResourceAccounting;
}

export interface VisualizationReservationRequest {
  owner: string;
  generation: string;
  priority: VisualizationResourcePriority;
  claims: readonly VisualizationResourceClaim[];
}

export interface VisualizationResourceCapacity {
  totalBytes: number | null;
  stageBytes?: Partial<Record<VisualizationResourceStage, number | null>>;
}

export interface VisualizationHardwareFacts {
  source: "tauri" | "browser" | "unknown";
  systemRamBytes: number | null;
  logicalCores: number | null;
}

export interface VisualizationReservation {
  id: number;
  owner: string;
  generation: string;
  priority: VisualizationResourcePriority;
  claims: readonly VisualizationResourceClaim[];
}

export type VisualizationReservationOutcome =
  | { ok: true; reservation: VisualizationReservation }
  | {
      ok: false;
      reason: string;
      requiredBytes: number;
      availableBytes: number | null;
      stage: VisualizationResourceStage | null;
    };

export interface VisualizationResourceSnapshot {
  readonly schema: typeof VISUALIZATION_RESOURCE_SCHEMA;
  readonly version: typeof VISUALIZATION_RESOURCE_VERSION;
  readonly sequence: number;
  readonly reservations: number;
  readonly allocations: number;
  readonly residentBytes: number;
  readonly reservedBytes: number;
  readonly committedBytes: number;
  readonly highWaterBytes: number;
  readonly residentHighWaterBytes: number;
  readonly reservedHighWaterBytes: number;
  readonly capacityBytes: number | null;
  readonly byStage: Readonly<Record<VisualizationResourceStage, Readonly<{
    allocations: number;
    residentBytes: number;
    reservedBytes: number;
    committedBytes: number;
    highWaterBytes: number;
    residentHighWaterBytes: number;
    reservedHighWaterBytes: number;
    capacityBytes: number | null;
  }>>>;
  readonly rejected: number;
  readonly released: number;
  readonly transitions: number;
  readonly prediction: Readonly<{
    samples: number;
    predictedBytes: number;
    actualBytes: number;
    underpredictions: number;
    maxUnderpredictionBytes: number;
    maxUtilization: number;
  }>;
  readonly hardware: Readonly<VisualizationHardwareFacts>;
}

interface OwnedAllocation {
  stage: VisualizationResourceStage;
  bytes: number;
  accounting: VisualizationResourceAccounting;
  reservations: Set<number>;
}

interface OwnedReservation {
  reservation: VisualizationReservation;
  claims: Map<string, VisualizationResourceClaim>;
}

type VisualizationResourceListener = () => void;

const STAGE_SET = new Set<string>(VISUALIZATION_RESOURCE_STAGES);

function emptyStageNumbers(): Record<VisualizationResourceStage, number> {
  return Object.fromEntries(VISUALIZATION_RESOURCE_STAGES.map((stage) => [stage, 0])) as
    Record<VisualizationResourceStage, number>;
}

function normalizedCapacity(capacity: VisualizationResourceCapacity): VisualizationResourceCapacity {
  const totalBytes = capacity.totalBytes == null
    ? null
    : finiteNonnegativeInteger(capacity.totalBytes, "visualization total capacity");
  const stageBytes: Partial<Record<VisualizationResourceStage, number | null>> = {};
  for (const stage of VISUALIZATION_RESOURCE_STAGES) {
    const value = capacity.stageBytes?.[stage];
    stageBytes[stage] = value == null
      ? null
      : finiteNonnegativeInteger(value, `${stage} capacity`);
  }
  return { totalBytes, stageBytes };
}

function finiteNonnegativeInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error(`${label} must be a nonnegative safe integer`);
  }
  return value;
}

function normalizeClaims(claims: readonly VisualizationResourceClaim[]): Map<string, VisualizationResourceClaim> {
  const normalized = new Map<string, VisualizationResourceClaim>();
  for (const claim of claims) {
    const allocationId = claim.allocationId.trim();
    if (!allocationId) throw new Error("visualization allocation id must not be empty");
    if (!STAGE_SET.has(claim.stage)) throw new Error(`unknown visualization resource stage '${claim.stage}'`);
    const bytes = finiteNonnegativeInteger(claim.bytes, `visualization allocation '${allocationId}' bytes`);
    const accounting = claim.accounting ?? "resident";
    if (accounting !== "resident" && accounting !== "reserved") {
      throw new Error(`visualization allocation '${allocationId}' has invalid accounting`);
    }
    const existing = normalized.get(allocationId);
    if (existing && (existing.stage !== claim.stage || existing.bytes !== bytes
      || (existing.accounting ?? "resident") !== accounting)) {
      throw new Error(`visualization allocation '${allocationId}' has conflicting claims`);
    }
    normalized.set(allocationId, { allocationId, stage: claim.stage, bytes, accounting });
  }
  return normalized;
}

export class VisualizationResourceGovernor {
  private capacity: VisualizationResourceCapacity;
  private readonly allocations = new Map<string, OwnedAllocation>();
  private readonly reservations = new Map<number, OwnedReservation>();
  private readonly listeners = new Set<VisualizationResourceListener>();
  private nextReservationId = 0;
  private sequence = 0;
  private residentBytes = 0;
  private reservedBytes = 0;
  private highWaterBytes = 0;
  private residentHighWaterBytes = 0;
  private reservedHighWaterBytes = 0;
  private readonly stageBytes = emptyStageNumbers();
  private readonly stageResidentBytes = emptyStageNumbers();
  private readonly stageReservedBytes = emptyStageNumbers();
  private readonly stageHighWater = emptyStageNumbers();
  private readonly stageResidentHighWater = emptyStageNumbers();
  private readonly stageReservedHighWater = emptyStageNumbers();
  private rejected = 0;
  private released = 0;
  private transitions = 0;
  private predictionSamples = 0;
  private predictedBytes = 0;
  private actualBytes = 0;
  private underpredictions = 0;
  private maxUnderpredictionBytes = 0;
  private maxUtilization = 0;
  private hardware: VisualizationHardwareFacts = {
    source: "unknown",
    systemRamBytes: null,
    logicalCores: null,
  };
  private snapshot: VisualizationResourceSnapshot;

  constructor(capacity: VisualizationResourceCapacity = { totalBytes: null }) {
    this.capacity = normalizedCapacity(capacity);
    this.snapshot = this.createSnapshot();
  }

  reserve(request: VisualizationReservationRequest): VisualizationReservationOutcome {
    const owner = request.owner.trim();
    const generation = request.generation.trim();
    if (!owner || !generation) throw new Error("visualization reservation owner and generation are required");
    const claims = normalizeClaims(request.claims);
    const admission = this.admission(claims, null);
    if (!admission.ok) {
      this.rejected += 1;
      this.publish();
      return admission;
    }

    const reservation: VisualizationReservation = Object.freeze({
      id: ++this.nextReservationId,
      owner,
      generation,
      priority: request.priority,
      claims: Object.freeze([...claims.values()].map((claim) => Object.freeze({ ...claim }))),
    });
    this.commitClaims(reservation.id, claims);
    this.reservations.set(reservation.id, { reservation, claims });
    this.updateHighWater();
    this.publish();
    return { ok: true, reservation };
  }

  transition(
    reservationId: number,
    claimsInput: readonly VisualizationResourceClaim[],
  ): VisualizationReservationOutcome {
    const current = this.reservations.get(reservationId);
    if (!current) throw new Error(`unknown visualization reservation ${reservationId}`);
    const claims = normalizeClaims(claimsInput);
    const admission = this.admission(claims, reservationId);
    if (!admission.ok) {
      this.rejected += 1;
      this.publish();
      return admission;
    }

    this.removeClaims(reservationId, current.claims);
    this.commitClaims(reservationId, claims);
    const reservation: VisualizationReservation = Object.freeze({
      ...current.reservation,
      claims: Object.freeze([...claims.values()].map((claim) => Object.freeze({ ...claim }))),
    });
    this.reservations.set(reservationId, { reservation, claims });
    this.transitions += 1;
    this.updateHighWater();
    this.publish();
    return { ok: true, reservation };
  }

  release(reservationId: number): boolean {
    const current = this.reservations.get(reservationId);
    if (!current) return false;
    this.removeClaims(reservationId, current.claims);
    this.reservations.delete(reservationId);
    this.released += 1;
    this.publish();
    return true;
  }

  setCapacity(capacity: VisualizationResourceCapacity): void {
    const normalized = normalizedCapacity(capacity);
    const total = normalized.totalBytes;
    if (total != null && this.residentBytes + this.reservedBytes > total) {
      throw new Error("visualization capacity cannot shrink below current residency");
    }
    for (const stage of VISUALIZATION_RESOURCE_STAGES) {
      const limit = normalized.stageBytes?.[stage];
      if (limit != null && this.stageBytes[stage] > limit) {
        throw new Error(`${stage} capacity cannot shrink below current residency`);
      }
    }
    this.capacity = normalized;
    this.publish();
  }

  recordPrediction(predictedBytes: number, actualBytes: number): void {
    const predicted = finiteNonnegativeInteger(predictedBytes, "predicted visualization bytes");
    const actual = finiteNonnegativeInteger(actualBytes, "actual visualization bytes");
    this.predictionSamples += 1;
    this.predictedBytes += predicted;
    this.actualBytes += actual;
    if (actual > predicted) {
      this.underpredictions += 1;
      this.maxUnderpredictionBytes = Math.max(this.maxUnderpredictionBytes, actual - predicted);
    }
    const utilization = predicted === 0 ? 0 : actual / predicted;
    this.maxUtilization = Math.max(this.maxUtilization, utilization);
    this.publish();
  }

  setHardwareFacts(facts: VisualizationHardwareFacts): void {
    const systemRamBytes = facts.systemRamBytes == null
      ? null
      : finiteNonnegativeInteger(facts.systemRamBytes, "system RAM bytes");
    const logicalCores = facts.logicalCores == null
      ? null
      : finiteNonnegativeInteger(facts.logicalCores, "logical core count");
    this.hardware = Object.freeze({
      source: facts.source,
      systemRamBytes,
      logicalCores,
    });
    this.publish();
  }

  stats(): VisualizationResourceSnapshot {
    return this.snapshot;
  }

  getReservation(id: number): VisualizationReservation | null {
    return this.reservations.get(id)?.reservation ?? null;
  }

  subscribe(listener: VisualizationResourceListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private admission(
    nextClaims: Map<string, VisualizationResourceClaim>,
    replacingReservationId: number | null,
  ): Exclude<VisualizationReservationOutcome, { ok: true }> | { ok: true } {
    let addedTotal = 0;
    const addedByStage = emptyStageNumbers();
    const removedByStage = emptyStageNumbers();
    let removedTotal = 0;
    const replacing = replacingReservationId == null
      ? null
      : this.reservations.get(replacingReservationId) ?? null;

    if (replacing) {
      for (const claim of replacing.claims.values()) {
        const allocation = this.allocations.get(claim.allocationId)!;
        if (allocation.reservations.size === 1) {
          removedTotal += allocation.bytes;
          removedByStage[allocation.stage] += allocation.bytes;
        }
      }
    }

    for (const claim of nextClaims.values()) {
      const allocation = this.allocations.get(claim.allocationId);
      const survivesReplacement = allocation && (
        replacingReservationId == null
        || allocation.reservations.size > 1
        || !replacing?.claims.has(claim.allocationId)
      );
      if (survivesReplacement) {
        if (allocation.stage !== claim.stage || allocation.bytes !== claim.bytes
          || allocation.accounting !== (claim.accounting ?? "resident")) {
          throw new Error(`visualization allocation '${claim.allocationId}' changed while still shared`);
        }
        continue;
      }
      addedTotal += claim.bytes;
      addedByStage[claim.stage] += claim.bytes;
    }

    const nextTotal = this.residentBytes + this.reservedBytes - removedTotal + addedTotal;
    const totalCapacity = this.capacity.totalBytes;
    if (totalCapacity != null && nextTotal > totalCapacity) {
      return {
        ok: false,
        reason: `Visualization needs ${nextTotal} bytes; aggregate capacity is ${totalCapacity}`,
        requiredBytes: nextTotal,
        availableBytes: totalCapacity,
        stage: null,
      };
    }
    for (const stage of VISUALIZATION_RESOURCE_STAGES) {
      const nextStage = this.stageBytes[stage] - removedByStage[stage] + addedByStage[stage];
      const capacity = this.capacity.stageBytes?.[stage] ?? null;
      if (capacity != null && nextStage > capacity) {
        return {
          ok: false,
          reason: `${stage} needs ${nextStage} bytes; stage capacity is ${capacity}`,
          requiredBytes: nextStage,
          availableBytes: capacity,
          stage,
        };
      }
    }
    return { ok: true };
  }

  private commitClaims(reservationId: number, claims: Map<string, VisualizationResourceClaim>): void {
    for (const claim of claims.values()) {
      const existing = this.allocations.get(claim.allocationId);
      if (existing) {
        if (existing.stage !== claim.stage || existing.bytes !== claim.bytes
          || existing.accounting !== (claim.accounting ?? "resident")) {
          throw new Error(`visualization allocation '${claim.allocationId}' changed while shared`);
        }
        existing.reservations.add(reservationId);
        continue;
      }
      this.allocations.set(claim.allocationId, {
        stage: claim.stage,
        bytes: claim.bytes,
        accounting: claim.accounting ?? "resident",
        reservations: new Set([reservationId]),
      });
      if ((claim.accounting ?? "resident") === "resident") {
        this.residentBytes += claim.bytes;
        this.stageResidentBytes[claim.stage] += claim.bytes;
      } else {
        this.reservedBytes += claim.bytes;
        this.stageReservedBytes[claim.stage] += claim.bytes;
      }
      this.stageBytes[claim.stage] += claim.bytes;
    }
  }

  private removeClaims(reservationId: number, claims: Map<string, VisualizationResourceClaim>): void {
    for (const claim of claims.values()) {
      const allocation = this.allocations.get(claim.allocationId);
      if (!allocation) continue;
      allocation.reservations.delete(reservationId);
      if (allocation.reservations.size > 0) continue;
      this.allocations.delete(claim.allocationId);
      if (allocation.accounting === "resident") {
        this.residentBytes -= allocation.bytes;
        this.stageResidentBytes[allocation.stage] -= allocation.bytes;
      } else {
        this.reservedBytes -= allocation.bytes;
        this.stageReservedBytes[allocation.stage] -= allocation.bytes;
      }
      this.stageBytes[allocation.stage] -= allocation.bytes;
    }
  }

  private updateHighWater(): void {
    this.highWaterBytes = Math.max(this.highWaterBytes, this.residentBytes + this.reservedBytes);
    this.residentHighWaterBytes = Math.max(this.residentHighWaterBytes, this.residentBytes);
    this.reservedHighWaterBytes = Math.max(this.reservedHighWaterBytes, this.reservedBytes);
    for (const stage of VISUALIZATION_RESOURCE_STAGES) {
      this.stageHighWater[stage] = Math.max(this.stageHighWater[stage], this.stageBytes[stage]);
      this.stageResidentHighWater[stage] = Math.max(
        this.stageResidentHighWater[stage],
        this.stageResidentBytes[stage],
      );
      this.stageReservedHighWater[stage] = Math.max(
        this.stageReservedHighWater[stage],
        this.stageReservedBytes[stage],
      );
    }
  }

  private createSnapshot(): VisualizationResourceSnapshot {
    const allocationCounts = emptyStageNumbers();
    for (const allocation of this.allocations.values()) allocationCounts[allocation.stage] += 1;
    const byStage = Object.fromEntries(VISUALIZATION_RESOURCE_STAGES.map((stage) => [stage, Object.freeze({
      allocations: allocationCounts[stage],
      residentBytes: this.stageResidentBytes[stage],
      reservedBytes: this.stageReservedBytes[stage],
      committedBytes: this.stageBytes[stage],
      highWaterBytes: this.stageHighWater[stage],
      residentHighWaterBytes: this.stageResidentHighWater[stage],
      reservedHighWaterBytes: this.stageReservedHighWater[stage],
      capacityBytes: this.capacity.stageBytes?.[stage] ?? null,
    })])) as Record<VisualizationResourceStage, Readonly<{
      allocations: number;
      residentBytes: number;
      reservedBytes: number;
      committedBytes: number;
      highWaterBytes: number;
      residentHighWaterBytes: number;
      reservedHighWaterBytes: number;
      capacityBytes: number | null;
    }>>;
    return Object.freeze({
      schema: VISUALIZATION_RESOURCE_SCHEMA,
      version: VISUALIZATION_RESOURCE_VERSION,
      sequence: this.sequence,
      reservations: this.reservations.size,
      allocations: this.allocations.size,
      residentBytes: this.residentBytes,
      reservedBytes: this.reservedBytes,
      committedBytes: this.residentBytes + this.reservedBytes,
      highWaterBytes: this.highWaterBytes,
      residentHighWaterBytes: this.residentHighWaterBytes,
      reservedHighWaterBytes: this.reservedHighWaterBytes,
      capacityBytes: this.capacity.totalBytes,
      byStage: Object.freeze(byStage),
      rejected: this.rejected,
      released: this.released,
      transitions: this.transitions,
      prediction: Object.freeze({
        samples: this.predictionSamples,
        predictedBytes: this.predictedBytes,
        actualBytes: this.actualBytes,
        underpredictions: this.underpredictions,
        maxUnderpredictionBytes: this.maxUnderpredictionBytes,
        maxUtilization: this.maxUtilization,
      }),
      hardware: Object.freeze({ ...this.hardware }),
    });
  }

  private publish(): void {
    this.sequence += 1;
    this.snapshot = this.createSnapshot();
    for (const listener of this.listeners) listener();
  }
}

let bufferSequence = 0;
const bufferIds = new WeakMap<object, number>();

export function visualizationBufferClaim(
  buffer: ArrayBufferLike,
  stage: VisualizationResourceStage,
): VisualizationResourceClaim {
  const key = buffer as object;
  let id = bufferIds.get(key);
  if (id == null) {
    id = ++bufferSequence;
    bufferIds.set(key, id);
  }
  return { allocationId: `buffer:${id}`, stage, bytes: buffer.byteLength };
}

export function visualizationReservationClaim(
  allocationId: string,
  stage: VisualizationResourceStage,
  bytes: number,
): VisualizationResourceClaim {
  return { allocationId, stage, bytes, accounting: "reserved" };
}

export const visualizationResourceGovernor = new VisualizationResourceGovernor();

export function getVisualizationResourceSnapshot(): VisualizationResourceSnapshot {
  return visualizationResourceGovernor.stats();
}

export function subscribeVisualizationResources(listener: VisualizationResourceListener): () => void {
  return visualizationResourceGovernor.subscribe(listener);
}

export function recordVisualizationTransientReservation(
  owner: string,
  generation: string,
  stage: VisualizationResourceStage,
  bytes: number,
): void {
  if (!(bytes > 0)) return;
  const resources = visualizationResourceGovernor.reserve({
    owner,
    generation,
    priority: "visible",
    claims: [visualizationReservationClaim(
      `transient:${owner}:${generation}`,
      stage,
      bytes,
    )],
  });
  if (resources.ok) visualizationResourceGovernor.release(resources.reservation.id);
}
