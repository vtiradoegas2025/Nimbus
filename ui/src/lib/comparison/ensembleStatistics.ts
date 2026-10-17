export interface EnsembleCellStatistics {
  mean: Float32Array;
  spread: Float32Array;
  probabilityPercent: Float32Array;
  completeCellCount: number;
  totalCellCount: number;
  memberCount: number;
  domainMeanSpread: number;
  maximumSpread: number;
  domainExceedancePercent: number;
}
const statisticsSources = new WeakMap<EnsembleCellStatistics, readonly ArrayLike<number>[]>();

/**
 * Exact per-cell ensemble statistics with a strict common finite sample.
 * Spread is the sample standard deviation (N−1); probability is the fraction
 * of all members strictly above the selected threshold.
 */
export function computeEnsembleCellStatistics(
  members: readonly ArrayLike<number>[],
  threshold: number,
): EnsembleCellStatistics {
  if (members.length < 2) throw new Error("Ensemble statistics require at least two members.");
  if (!Number.isFinite(threshold)) throw new Error("Probability threshold must be finite.");
  const length = members[0].length;
  if (members.some((member) => member.length !== length)) throw new Error("Ensemble member slices changed shape after alignment.");
  const mean = new Float32Array(length);
  const spread = new Float32Array(length);
  const probabilityPercent = new Float32Array(length);
  mean.fill(Number.NaN);
  spread.fill(Number.NaN);
  probabilityPercent.fill(Number.NaN);
  let completeCellCount = 0;
  let spreadSum = 0;
  let maximumSpread = 0;
  let exceedanceSum = 0;
  for (let index = 0; index < length; index += 1) {
    let cellMean = 0;
    let m2 = 0;
    let exceedances = 0;
    let complete = true;
    for (let memberIndex = 0; memberIndex < members.length; memberIndex += 1) {
      const value = members[memberIndex][index];
      if (!Number.isFinite(value)) { complete = false; break; }
      const count = memberIndex + 1;
      const delta = value - cellMean;
      cellMean += delta / count;
      m2 += delta * (value - cellMean);
      if (value > threshold) exceedances += 1;
    }
    if (!complete) continue;
    const cellSpread = Math.sqrt(m2 / (members.length - 1));
    const probability = exceedances / members.length * 100;
    mean[index] = cellMean;
    spread[index] = cellSpread;
    probabilityPercent[index] = probability;
    completeCellCount += 1;
    spreadSum += cellSpread;
    maximumSpread = Math.max(maximumSpread, cellSpread);
    exceedanceSum += probability;
  }
  if (completeCellCount === 0) throw new Error("No cell is finite across every ensemble member.");
  const result = {
    mean,
    spread,
    probabilityPercent,
    completeCellCount,
    totalCellCount: length,
    memberCount: members.length,
    domainMeanSpread: spreadSum / completeCellCount,
    maximumSpread,
    domainExceedancePercent: exceedanceSum / completeCellCount,
  };
  statisticsSources.set(result, [...members]);
  return result;
}

export function withEnsembleThreshold(
  members: readonly ArrayLike<number>[],
  moments: EnsembleCellStatistics,
  threshold: number,
): EnsembleCellStatistics {
  if (!Number.isFinite(threshold)) throw new Error("Probability threshold must be finite.");
  const sources = statisticsSources.get(moments);
  if (members.length !== moments.memberCount || members.length < 2
      || !sources || members.some((member, index) => member !== sources[index] || member.length !== moments.totalCellCount)) {
    throw new Error("Threshold statistics require the same completed ensemble slices.");
  }
  const probabilityPercent = new Float32Array(moments.totalCellCount);
  probabilityPercent.fill(Number.NaN);
  let completeCellCount = 0;
  let exceedanceSum = 0;
  for (let index = 0; index < probabilityPercent.length; index++) {
    let exceedances = 0;
    let complete = true;
    for (const member of members) {
      const value = member[index];
      if (!Number.isFinite(value)) { complete = false; break; }
      if (value > threshold) exceedances++;
    }
    if (!complete) continue;
    const probability = exceedances / members.length * 100;
    probabilityPercent[index] = probability;
    completeCellCount++;
    exceedanceSum += probability;
  }
  if (completeCellCount !== moments.completeCellCount) throw new Error("Completed ensemble support changed during threshold evaluation.");
  const result = { ...moments, probabilityPercent, domainExceedancePercent: exceedanceSum / completeCellCount };
  statisticsSources.set(result, sources);
  return result;
}
