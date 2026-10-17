/** Bounded-concurrency task pool. Failures are delivered to onSettled, never thrown;
 *  returning false from onSettled stops the pool without racing remaining callbacks. */

export type PoolResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: unknown };

/**
 * Process `count` indices through `task` with at most `concurrency` running
 * concurrently. Returns once every index has settled or the pool was stopped.
 */
export async function mapPool<T>(
  count: number,
  concurrency: number,
  task: (index: number) => Promise<T>,
  onSettled: (index: number, result: PoolResult<T>) => boolean,
): Promise<void> {
  if (count <= 0) return;
  const limit = Math.max(1, Math.min(Math.floor(concurrency), count));
  let next = 0;
  let stopped = false;

  async function worker(): Promise<void> {
    for (;;) {
      if (stopped) return;
      const i = next;
      if (i >= count) return;
      next = i + 1;
      let result: PoolResult<T>;
      try {
        result = { ok: true, value: await task(i) };
      } catch (error) {
        result = { ok: false, error };
      }
      // Pool may have been stopped while we awaited; drop the settle.
      if (stopped) return;
      if (!onSettled(i, result)) {
        stopped = true;
        return;
      }
    }
  }

  await Promise.all(Array.from({ length: limit }, () => worker()));
}
