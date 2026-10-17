import {
  visualizationReservationClaim,
  visualizationBufferClaim,
  visualizationResourceGovernor,
  type VisualizationResourceClaim,
} from "@/lib/core/visualizationResourceGovernor";

export interface VtkWorkerDatasetBuffer {
  buffer: ArrayBufferLike;
  sharedWithMain: boolean;
}

/** Tracks immutable prepared lattices retained by one extraction worker.
 * Replacement keeps both generations accounted until the worker replies: the
 * old lattice is deleted while processing the new request, not when the UI
 * posts it. */
export class VtkWorkerDatasetOwnership {
  private readonly owner: string;
  private currentReservationId: number | null = null;
  private readonly pending = new Map<number, number>();

  constructor(owner: string) {
    const normalized = owner.trim();
    if (!normalized) throw new Error("VTK worker dataset owner is required");
    this.owner = normalized;
  }

  begin(requestId: number, buffers: readonly VtkWorkerDatasetBuffer[]): void {
    if (!Number.isSafeInteger(requestId) || requestId < 1) {
      throw new Error("VTK worker dataset request id must be a positive safe integer");
    }
    this.discard(requestId);
    const claims: VisualizationResourceClaim[] = buffers.map(({ buffer, sharedWithMain }) =>
      visualizationBufferClaim(buffer, sharedWithMain ? "prepared-main" : "worker-resident")
    );
    const outcome = visualizationResourceGovernor.reserve({
      owner: this.owner,
      generation: `request:${requestId}`,
      priority: "visible",
      claims,
    });
    if (!outcome.ok) throw new Error(outcome.reason);
    this.pending.set(requestId, outcome.reservation.id);
  }

  reserveCopy(requestId: number, bytes: number): void {
    if (!Number.isSafeInteger(requestId) || requestId < 1) {
      throw new Error("VTK worker dataset request id must be a positive safe integer");
    }
    if (!Number.isSafeInteger(bytes) || bytes < 0) {
      throw new Error("VTK worker dataset copy bytes must be a non-negative safe integer");
    }
    this.discard(requestId);
    const outcome = visualizationResourceGovernor.reserve({
      owner: this.owner,
      generation: `request:${requestId}:copy-reservation`,
      priority: "visible",
      claims: bytes === 0 ? [] : [visualizationReservationClaim(
        `${this.owner}:request:${requestId}:copy`,
        "worker-resident",
        bytes,
      )],
    });
    if (!outcome.ok) throw new Error(outcome.reason);
    this.pending.set(requestId, outcome.reservation.id);
  }

  materialize(requestId: number, buffers: readonly VtkWorkerDatasetBuffer[]): void {
    const reservationId = this.pending.get(requestId);
    if (reservationId == null) {
      this.begin(requestId, buffers);
      return;
    }
    const claims: VisualizationResourceClaim[] = buffers.map(({ buffer, sharedWithMain }) =>
      visualizationBufferClaim(buffer, sharedWithMain ? "prepared-main" : "worker-resident")
    );
    const outcome = visualizationResourceGovernor.transition(reservationId, claims);
    if (outcome.ok) return;
    this.pending.delete(requestId);
    visualizationResourceGovernor.release(reservationId);
    throw new Error(outcome.reason);
  }

  commit(requestId: number): void {
    const next = this.pending.get(requestId);
    if (next == null) return;
    this.pending.delete(requestId);
    if (this.currentReservationId != null) {
      visualizationResourceGovernor.release(this.currentReservationId);
    }
    this.currentReservationId = next;
  }

  discard(requestId: number): void {
    const reservation = this.pending.get(requestId);
    if (reservation == null) return;
    this.pending.delete(requestId);
    visualizationResourceGovernor.release(reservation);
  }

  reset(): void {
    if (this.currentReservationId != null) {
      visualizationResourceGovernor.release(this.currentReservationId);
      this.currentReservationId = null;
    }
    for (const reservation of this.pending.values()) {
      visualizationResourceGovernor.release(reservation);
    }
    this.pending.clear();
  }
}
