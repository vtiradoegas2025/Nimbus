import type { PythonNumericArray } from "./pythonNumericArray";

export const MAX_PYTHON_ARRAY_EXPORT_BYTES = 64 * 1024 * 1024;
export const MAX_PYTHON_ENCODED_EXPORT_BYTES = MAX_PYTHON_ARRAY_EXPORT_BYTES + 4096;

export interface PythonArrayExportRequest {
  id: number;
  data: PythonNumericArray;
  shape: readonly number[];
  dtype: string;
  format: "npy" | "csv";
}

export type PythonArrayExportReply = { id: number; bytes: Uint8Array } | { id: number; error: string };
