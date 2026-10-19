export function investigationHistoryAxis(steps: readonly { index: number; simulationTimeS: number | null }[]): {
  values: number[]; unit: "s" | "index"; title: string;
} {
  const physical = steps.every(step => step.simulationTimeS != null && Number.isFinite(step.simulationTimeS));
  return physical
    ? { values: steps.map(step => step.simulationTimeS!), unit: "s", title: "simulation time (s)" }
    : { values: steps.map(step => step.index), unit: "index", title: "source frame index · physical time unavailable" };
}

export class InvestigationHistoryOwner {
  private identity = "";
  private generation = 0;
  private controller: AbortController | null = null;
  select(identity: string): void {
    if (this.identity === identity) return;
    this.identity = identity;
    this.generation++;
  }
  begin(): { identity: string; generation: number; signal: AbortSignal } {
    this.cancel();
    this.controller = new AbortController();
    return { identity: this.identity, generation: this.generation, signal: this.controller.signal };
  }
  current(request: { identity: string; generation: number; signal: AbortSignal }): boolean {
    return request.identity === this.identity && request.generation === this.generation && !request.signal.aborted;
  }
  cancel(): void {
    this.generation++;
    this.controller?.abort();
    this.controller = null;
  }
}
