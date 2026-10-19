export const PACKAGED_WORKER_WATCHDOG_STAGES = [
  ...[1, 2].flatMap(cycle => [
    `cycle-${cycle}-scalar-preparation`,
    `cycle-${cycle}-contour-extraction`,
    `cycle-${cycle}-vector-preparation`,
    `cycle-${cycle}-streamline-extraction`,
    `cycle-${cycle}-worker-shutdown`,
    `cycle-${cycle}-retirement-checks`,
  ]),
  "packaged-native-bricks",
] as const;

export function workerWatchdogMaximum(stages: Readonly<Record<string, number>>): number {
  return Math.max(...PACKAGED_WORKER_WATCHDOG_STAGES.map(stage => {
    const delay = stages[stage];
    if (!Number.isFinite(delay) || delay < 0) throw new Error(`Missing or invalid worker watchdog stage: ${stage}`);
    return delay;
  }));
}
