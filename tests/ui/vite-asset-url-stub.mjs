/*
 * Stands in for Vite's `?worker&url` (and friends) under plain node.
 *
 * `pyodideRuntime.ts` imports the interpreter worker's built URL as
 * `./pyodideWorker.ts?worker&url`, which is how Vite is told to emit the worker
 * as its own chunk and hand back its address — the only form that survives
 * appending a cache-busting query, which WebKit needs to reload a worker at all
 * (see `ensureWorker`). Node has no such concept: it strips nothing, resolves
 * the query away, and tries to execute the worker module on the main thread.
 *
 * The node suites reach `pyodideRuntime` only for its re-exported package-table
 * lookups and its pure helpers; none of them construct a worker, so the address
 * is never read. It is deliberately not a plausible URL — anything that did try
 * to load it should fail loudly here rather than quietly fetch something.
 */
export default "vite-asset-url-unavailable-outside-vite";
