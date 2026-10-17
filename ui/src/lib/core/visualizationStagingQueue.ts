import {
  visualizationResourceGovernor,
  type VisualizationResourceGovernor,
  type VisualizationReservationOutcome,
  type VisualizationReservationRequest,
} from "./visualizationResourceGovernor";

interface PendingReservation {
  request: VisualizationReservationRequest;
  bytes: number;
  granted: (outcome: VisualizationReservationOutcome) => void;
}

export class VisualizationStagingQueue {
  private readonly pending: PendingReservation[] = [];
  private readonly active = new Set<number>();
  private unsubscribe: (() => void) | null = null;
  private draining = false;
  private readonly governor: VisualizationResourceGovernor;
  private maximumBytes: number;

  constructor(
    governor: VisualizationResourceGovernor,
    maximumBytes = 96 * 1024 * 1024,
  ) {
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes <= 0) {
      throw new Error("Visualization staging limit must be a positive safe integer");
    }
    this.governor = governor;
    this.maximumBytes = maximumBytes;
  }

  schedule(
    request: VisualizationReservationRequest,
    granted: PendingReservation["granted"],
  ): () => void {
    const bytes = request.claims.reduce((total, claim) =>
      total + (claim.accounting === "reserved" ? claim.bytes : 0), 0);
    if (!Number.isSafeInteger(bytes) || bytes < 0) throw new Error("Invalid visualization staging bytes");
    const pending = { request, bytes, granted };
    this.pending.push(pending);
    this.unsubscribe ??= this.governor.subscribe(() => queueMicrotask(() => this.drain()));
    this.drain();
    return () => {
      const index = this.pending.indexOf(pending);
      if (index >= 0) this.pending.splice(index, 1);
      this.drain();
    };
  }

  setMaximumBytes(maximumBytes: number): void {
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes <= 0) {
      throw new Error("Visualization staging limit must be a positive safe integer");
    }
    this.maximumBytes = maximumBytes;
    this.drain();
  }

  private drain(): void {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.pending.length > 0) {
        const priorities = { interactive: 0, visible: 1, playback: 2, prefetch: 3, background: 4 };
        let nextIndex = 0;
        for (let index = 1; index < this.pending.length; index++) {
          if (priorities[this.pending[index].request.priority]
              < priorities[this.pending[nextIndex].request.priority]) nextIndex = index;
        }
        const pending = this.pending[nextIndex];
        let reserved = 0;
        const allocations = new Set<string>();
        for (const id of this.active) {
          const reservation = this.governor.getReservation(id);
          if (!reservation) { this.active.delete(id); continue; }
          for (const claim of reservation.claims) {
            if (claim.accounting !== "reserved" || allocations.has(claim.allocationId)) continue;
            allocations.add(claim.allocationId);
            reserved += claim.bytes;
          }
        }
        // A single oversized field can progress without overlapping other staging.
        if (reserved > 0 && reserved + pending.bytes > this.maximumBytes) break;
        this.pending.splice(nextIndex, 1);
        const outcome = this.governor.reserve(pending.request);
        if (outcome.ok) this.active.add(outcome.reservation.id);
        pending.granted(outcome);
      }
    } finally {
      this.draining = false;
      if (this.pending.length === 0) {
        this.unsubscribe?.();
        this.unsubscribe = null;
      }
    }
  }
}

export const visualizationStagingQueue = new VisualizationStagingQueue(visualizationResourceGovernor);
