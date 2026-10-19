export interface ScalarVolumeStats {
  min: number;
  max: number;
  p2: number;
  p98: number;
  p995: number;
  positiveP98: number;
  negativeP02: number;
}

interface ScalarSamples {
  values: ArrayLike<number>;
  validity?: ArrayLike<number>;
}

export function scalarVolumeStats(
  values: ArrayLike<number>,
  validity?: ArrayLike<number>,
): ScalarVolumeStats {
  return scalarVolumeBatchStats([{ values, validity }]);
}

export function scalarVolumeBatchStats(inputs: readonly ScalarSamples[]): ScalarVolumeStats {
  const total = inputs.reduce((sum, input) => sum + input.values.length, 0);
  const stride = Math.max(1, Math.floor(total / 200_000));
  const samples: number[] = [];
  let minimum = Infinity;
  let maximum = -Infinity;
  let linear = 0;
  for (const { values, validity } of inputs) {
    if (validity && validity.length !== values.length) throw new Error("Scalar statistics validity length mismatch");
    for (let index = 0; index < values.length; index++, linear++) {
      if (validity?.[index] === 0) continue;
      const value = values[index];
      if (!Number.isFinite(value)) continue;
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
      if (linear % stride === 0) samples.push(value);
    }
  }
  samples.sort((left, right) => left - right);
  const percentile = (fraction: number, first = 0, end = samples.length): number =>
    end > first ? samples[first + Math.round(fraction * (end - first - 1))] : Number.NaN;
  let firstNonnegative = 0;
  while (firstNonnegative < samples.length && samples[firstNonnegative] < 0) firstNonnegative++;
  let firstPositive = firstNonnegative;
  while (firstPositive < samples.length && samples[firstPositive] === 0) firstPositive++;
  return {
    min: Number.isFinite(minimum) ? minimum : 0,
    max: Number.isFinite(maximum) ? maximum : 0,
    p2: percentile(0.02),
    p98: percentile(0.98),
    p995: percentile(0.995),
    // If a rare sign falls between strided samples, retain its exact extremum.
    positiveP98: firstPositive < samples.length ? percentile(0.98, firstPositive) : Math.max(0, maximum),
    negativeP02: firstNonnegative > 0 ? percentile(0.02, 0, firstNonnegative) : Math.min(0, minimum),
  };
}
