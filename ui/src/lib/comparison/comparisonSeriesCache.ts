export interface ComparisonFrameIdentity {
  datasetIdentity: string;
  runDir: string;
  publicationGeneration: string | null;
  stepIndex: number;
  fieldDefinitions?: readonly string[];
  fieldUnits?: readonly (string | null)[];
}

export type ComparisonSeriesAnalysisIdentity =
  | {
      kind: "scalar";
      field: string;
      rank: 2 | 3;
      leftGeometry: unknown;
      rightGeometry: unknown;
    }
  | {
      kind: "vector";
      components: readonly [string, string];
      region: {
        lower: readonly [number, number, number];
        upper: readonly [number, number, number];
      };
      leftGeometry: unknown;
      rightGeometry: unknown;
    };

export function comparisonSeriesMatchKey(input: {
  left: ComparisonFrameIdentity;
  right: ComparisonFrameIdentity;
  simulationTimeS: number;
  analysis: ComparisonSeriesAnalysisIdentity;
}): string {
  return JSON.stringify([
    "nimbus.comparison-series-match",
    1,
    input.left.datasetIdentity,
    input.left.runDir,
    input.left.publicationGeneration,
    input.left.stepIndex,
    input.left.fieldDefinitions ?? [],
    input.left.fieldUnits ?? [],
    input.right.datasetIdentity,
    input.right.runDir,
    input.right.publicationGeneration,
    input.right.stepIndex,
    input.right.fieldDefinitions ?? [],
    input.right.fieldUnits ?? [],
    input.simulationTimeS,
    input.analysis,
  ]);
}

export interface ComparisonSeriesRequest<TRequest> {
  key: string;
  request: TRequest;
}

interface CacheRun<TValue> {
  token: number;
  prefix: TValue[];
}

/**
 * Retains the exact completed prefix of one ordered matched-time series.
 * A changed match keeps the preceding prefix and invalidates that match and
 * every alignment-dependent row after it. Appends therefore start at the
 * first newly matched frame.
 */
export class OrderedComparisonSeriesCache<TValue> {
  private token = 0;
  private keys: string[] = [];
  private values: TValue[] = [];

  begin(nextKeys: readonly string[]): CacheRun<TValue> {
    this.token += 1;
    let prefixLength = 0;
    const limit = Math.min(this.keys.length, nextKeys.length, this.values.length);
    while (prefixLength < limit && this.keys[prefixLength] === nextKeys[prefixLength]) {
      prefixLength += 1;
    }
    this.keys = [...nextKeys];
    this.values = this.values.slice(0, prefixLength);
    return { token: this.token, prefix: [...this.values] };
  }

  commit(token: number, index: number, key: string, value: TValue): boolean {
    if (token !== this.token || index !== this.values.length || this.keys[index] !== key) {
      return false;
    }
    this.values.push(value);
    return true;
  }

  snapshot(token: number): TValue[] | null {
    return token === this.token ? [...this.values] : null;
  }

  matchingPrefix(nextKeys: readonly string[]): TValue[] {
    let length = 0;
    while (length < this.values.length && length < nextKeys.length && this.keys[length] === nextKeys[length]) length++;
    return this.values.slice(0, length);
  }
}

export async function buildIncrementalComparisonSeries<TRequest, TValue>(input: {
  cache: OrderedComparisonSeriesCache<TValue>;
  requests: readonly ComparisonSeriesRequest<TRequest>[];
  load: (request: TRequest, index: number) => Promise<TValue>;
  cancelled?: () => boolean;
  onProgress?: (values: readonly TValue[]) => void;
}): Promise<TValue[]> {
  const run = input.cache.begin(input.requests.map(({ key }) => key));
  let values = run.prefix;
  input.onProgress?.(values);
  for (let index = values.length; index < input.requests.length; index += 1) {
    if (input.cancelled?.()) throw new DOMException("Comparison series cancelled", "AbortError");
    const item = input.requests[index];
    const value = await input.load(item.request, index);
    if (input.cancelled?.() || !input.cache.commit(run.token, index, item.key, value)) {
      throw new DOMException("Comparison series cancelled", "AbortError");
    }
    values = input.cache.snapshot(run.token) ?? [];
    input.onProgress?.(values);
  }
  return values;
}
