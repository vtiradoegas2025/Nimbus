/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL the Pyodide runtime is fetched from. Unset in the app build, which
   * falls back to the pinned CDN release; the browser-test config points it at
   * a locally vendored copy so the Python tab can be exercised end to end
   * without a network round trip on every run.
   */
  readonly VITE_PYODIDE_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
