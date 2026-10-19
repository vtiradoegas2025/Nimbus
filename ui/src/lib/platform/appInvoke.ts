/*
 * Single entry point for all Tauri commands.
 * Normalises Rust's `{ code, message }` rejection shape into AppInvokeError.
 * Branch on `code`, never on message text — messages are prose and change.
 */

/** Mirrors `AppError::code` in `src-tauri/src/error.rs`. */
export const APP_ERROR_CODES = [
  "io",
  "not_found",
  "invalid",
  "subprocess",
  "unknown_run",
  "task",
] as const;

export type AppErrorCode = (typeof APP_ERROR_CODES)[number];

function isAppErrorCode(value: unknown): value is AppErrorCode {
  return typeof value === "string" && (APP_ERROR_CODES as readonly string[]).includes(value);
}

export class AppInvokeError extends Error {
  /** Null when the rejection did not come from `AppError`, so callers can fail closed. */
  readonly code: AppErrorCode | null;
  readonly command: string;

  constructor(command: string, message: string, code: AppErrorCode | null) {
    super(message);
    this.name = "AppInvokeError";
    this.command = command;
    this.code = code;
  }
}

/** Normalize any rejection shape into an AppInvokeError without losing text. */
export function toAppInvokeError(command: string, raw: unknown): AppInvokeError {
  if (raw instanceof AppInvokeError) return raw;
  if (typeof raw === "string") return new AppInvokeError(command, raw, null);
  if (raw && typeof raw === "object") {
    const record = raw as { message?: unknown; code?: unknown };
    const code = isAppErrorCode(record.code) ? record.code : null;
    if (typeof record.message === "string") {
      return new AppInvokeError(command, record.message, code);
    }
    if (raw instanceof Error) return new AppInvokeError(command, raw.message, code);
  }
  return new AppInvokeError(command, `${command} failed: ${String(raw)}`, null);
}

export function appErrorCode(error: unknown): AppErrorCode | null {
  return error instanceof AppInvokeError ? error.code : null;
}

/** True for transient conditions (mid-publish, dir not yet created). Other failures repeat on retry. */
export function isTransientAppError(error: unknown): boolean {
  const code = appErrorCode(error);
  return code === "io" || code === "not_found";
}

export async function invokeCommand<T>(
  command: string,
  args?: Record<string, unknown>,
): Promise<T> {
  const { invoke } = await import("@tauri-apps/api/core");
  try {
    return await invoke<T>(command, args);
  } catch (raw) {
    throw toAppInvokeError(command, raw);
  }
}
