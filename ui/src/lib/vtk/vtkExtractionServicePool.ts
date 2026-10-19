export interface VtkExtractionServicePoolSnapshot {
  readonly warmServices: number;
  readonly maximumWarmServices: number;
  readonly evictions: number;
}

interface WarmService {
  retire: () => void;
}
type PoolListener = () => void;

export const MAX_WARM_EXTRACTION_SERVICES = 3;

export class VtkExtractionServicePool {
  private readonly maximumWarmServices: number;
  private readonly warm = new Map<object, WarmService>();
  private readonly listeners = new Set<PoolListener>();
  private evictions = 0;

  constructor(maximumWarmServices = MAX_WARM_EXTRACTION_SERVICES) {
    if (!Number.isSafeInteger(maximumWarmServices) || maximumWarmServices < 0) {
      throw new Error("Warm extraction service limit must be a non-negative safe integer");
    }
    this.maximumWarmServices = maximumWarmServices;
  }

  activate(owner: object): void {
    if (this.warm.delete(owner)) this.publish();
  }

  park(owner: object, retire: () => void): void {
    this.warm.delete(owner);
    this.warm.set(owner, { retire });
    while (this.warm.size > this.maximumWarmServices) {
      const oldest = this.warm.entries().next().value as [object, WarmService] | undefined;
      if (!oldest) break;
      this.warm.delete(oldest[0]);
      this.evictions += 1;
      oldest[1].retire();
    }
    this.publish();
  }

  remove(owner: object): void {
    if (this.warm.delete(owner)) this.publish();
  }

  snapshot(): VtkExtractionServicePoolSnapshot {
    return Object.freeze({
      warmServices: this.warm.size,
      maximumWarmServices: this.maximumWarmServices,
      evictions: this.evictions,
    });
  }

  reset(): void {
    const services = [...this.warm.values()];
    this.warm.clear();
    for (const service of services) service.retire();
    this.publish();
  }

  subscribe(listener: PoolListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private publish(): void {
    for (const listener of this.listeners) listener();
  }
}

export const vtkExtractionServicePool = new VtkExtractionServicePool();

export function subscribeVtkExtractionServicePool(listener: PoolListener): () => void {
  return vtkExtractionServicePool.subscribe(listener);
}
