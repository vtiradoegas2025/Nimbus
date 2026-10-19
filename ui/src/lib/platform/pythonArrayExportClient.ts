import pythonArrayExportWorkerUrl from "./pythonArrayExport.worker.ts?worker&url";
import { MAX_CSV_CELLS, validatePythonNumericArray } from "./pythonExport";
import { decodePythonNumericArray, pythonArrayDtype, type PythonNumericArray } from "./pythonNumericArray";
import { MAX_PYTHON_ARRAY_EXPORT_BYTES, MAX_PYTHON_ENCODED_EXPORT_BYTES, type PythonArrayExportRequest, type PythonArrayExportReply } from "./pythonArrayExportProtocol";

export interface PythonArrayExportWorker {
  onmessage: ((event: MessageEvent<PythonArrayExportReply>) => void) | null;
  onerror: ((event: ErrorEvent) => void) | null;
  onmessageerror: ((event: MessageEvent) => void) | null;
  postMessage(request: PythonArrayExportRequest): void;
  terminate(): void;
}

const abortError = () => new DOMException("Python array export cancelled", "AbortError");
const workerSession = Date.now().toString(36);
let workerGeneration = 0;

function createDefaultWorker(): PythonArrayExportWorker {
  const separator = pythonArrayExportWorkerUrl.includes("?") ? "&" : "?";
  return new Worker(`${pythonArrayExportWorkerUrl}${separator}session=${workerSession}&generation=${++workerGeneration}`, {
    type: "module", name: "nimbus-python-array-export",
  });
}

export class PythonArrayExportService {
  private active: symbol | null = null;
  private sequence = 0;
  private readonly createWorker: () => PythonArrayExportWorker;

  constructor(createWorker: () => PythonArrayExportWorker = createDefaultWorker) {
    this.createWorker = createWorker;
  }

  encode(data: PythonNumericArray, shape: readonly number[], format: "npy" | "csv", dtype = pythonArrayDtype(data), signal?: AbortSignal): Promise<Uint8Array> {
    if (signal?.aborted) return Promise.reject(abortError());
    if (this.active != null) return Promise.reject(new Error("Another Python array export is still encoding; wait or cancel it first"));
    try {
      validatePythonNumericArray(data, shape, dtype, false);
      if (data.byteLength > MAX_PYTHON_ARRAY_EXPORT_BYTES) throw new Error("Python array export exceeds its 64 MiB input limit");
      if (format === "csv" && (data.length > MAX_CSV_CELLS || shape.length < 1 || shape.length > 2)) throw new Error("CSV requires a 1D or 2D array of at most 1,000,000 values; save as .npy instead");
    } catch (error) { return Promise.reject(error); }
    const token = Symbol("python-array-export");
    this.active = token;
    const id = ++this.sequence;
    return new Promise((resolve, reject) => {
      let worker: PythonArrayExportWorker | null = null;
      let timer: ReturnType<typeof setTimeout> | null = null;
      let settled = false;
      const finish = (error: unknown, bytes?: Uint8Array) => {
        if (settled) return;
        settled = true;
        signal?.removeEventListener("abort", abort);
        if (timer != null) clearTimeout(timer);
        worker?.terminate();
        if (this.active === token) this.active = null;
        if (error != null) reject(error);
        else resolve(bytes!);
      };
      const abort = () => finish(abortError());
      signal?.addEventListener("abort", abort, { once: true });
      try {
        worker = this.createWorker();
        worker.onmessage = event => {
          if (event.data.id !== id) { finish(new Error("Python array export reply identity mismatch")); return; }
          if ("error" in event.data) { finish(new Error(event.data.error)); return; }
          const bytes = event.data.bytes;
          if (!(bytes instanceof Uint8Array) || bytes.byteLength > MAX_PYTHON_ENCODED_EXPORT_BYTES) {
            finish(new Error("Python array export returned an invalid encoded buffer")); return;
          }
          finish(null, bytes);
        };
        worker.onerror = event => finish(new Error(event.message || "Python array export worker failed"));
        worker.onmessageerror = () => finish(new Error("Python array export reply could not be decoded"));
        timer = setTimeout(() => finish(new Error("Python array export exceeded its 30 second encoding limit")), 30_000);
        const snapshot = data.buffer instanceof ArrayBuffer && data.byteOffset === 0
          && data.byteLength === data.buffer.byteLength ? data
          : decodePythonNumericArray(new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(), dtype);
        worker.postMessage({ id, data: snapshot, shape: [...shape], dtype, format });
      } catch (error) { finish(error); }
    });
  }
}

export const pythonArrayExportService = new PythonArrayExportService();
