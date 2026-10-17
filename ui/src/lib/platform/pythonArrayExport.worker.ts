import { encodeCsv, encodeNpy, MAX_CSV_CELLS } from "./pythonExport";
import { MAX_PYTHON_ARRAY_EXPORT_BYTES, MAX_PYTHON_ENCODED_EXPORT_BYTES, type PythonArrayExportRequest, type PythonArrayExportReply } from "./pythonArrayExportProtocol";

const scope = self as unknown as {
  addEventListener(type: "message", handler: (event: MessageEvent<PythonArrayExportRequest>) => void): void;
  postMessage(reply: PythonArrayExportReply, transfer?: Transferable[]): void;
};
scope.addEventListener("message", event => {
  const request = event.data;
  try {
    if (!Number.isSafeInteger(request.id) || request.id < 1 || !["npy", "csv"].includes(request.format)
        || request.data.byteLength > MAX_PYTHON_ARRAY_EXPORT_BYTES) throw new Error("Python array export request exceeds its contract");
    if (request.format === "csv" && request.data.length > MAX_CSV_CELLS) throw new Error("CSV export exceeds its value limit");
    const bytes = request.format === "npy" ? encodeNpy(request.data, request.shape, request.dtype)
      : new TextEncoder().encode(encodeCsv(request.data, request.shape, request.dtype));
    if (bytes.byteLength > MAX_PYTHON_ENCODED_EXPORT_BYTES) throw new Error("Python encoded export exceeds its output limit");
    scope.postMessage({ id: request.id, bytes }, [bytes.buffer as ArrayBuffer]);
  } catch (error) {
    scope.postMessage({ id: request.id, error: error instanceof Error ? error.message : String(error) });
  }
});
