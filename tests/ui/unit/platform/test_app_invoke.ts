/*
 * IPC error-contract tests.
 * Run via: npm run test:app-invoke
 *
 * Tauri rejects with the serialized Rust error verbatim — a plain object, not
 * an Error. Before normalization every caller's
 * `e instanceof Error ? e.message : String(e)` rendered "[object Object]",
 * and retry loops could not tell a transient miss from a permanent one.
 */

import {
  APP_ERROR_CODES,
  AppInvokeError,
  appErrorCode,
  isTransientAppError,
  toAppInvokeError,
} from "../../../../ui/src/lib/platform/appInvoke.ts";

let passed = 0;
const failures: string[] = [];
const check = (condition: boolean, label: string) => condition ? passed += 1 : failures.push(label);

// Regression: the exact shape Tauri rejects with for AppError.
const wire = toAppInvokeError("read_field_bytes", { code: "not_found", message: "path not found: /runs/step_000001/w.npy" });
check(wire instanceof Error, "a wire rejection normalizes to a real Error");
check(wire.message === "path not found: /runs/step_000001/w.npy", "the Rust message survives normalization");
check(wire.code === "not_found", "the Rust code survives normalization");
check(wire.command === "read_field_bytes", "the failing command is retained");

// The bug this prevents: callers stringify with `String(e)` on a non-Error.
check(String({ message: "x" }) === "[object Object]", "a raw wire object stringifies uselessly");
check(
  (wire instanceof Error ? wire.message : String(wire)) === "path not found: /runs/step_000001/w.npy",
  "the standard caller idiom now yields the real message",
);

// Unknown / non-AppError shapes must not be silently mistaken for a code.
check(toAppInvokeError("c", { code: "teapot", message: "m" }).code === null, "an unrecognized code is rejected, not trusted");
check(toAppInvokeError("c", "plain string failure").message === "plain string failure", "a string rejection is preserved");
check(toAppInvokeError("c", new Error("boom")).message === "boom", "an Error rejection is preserved");
check(toAppInvokeError("c", undefined).code === null, "an opaque rejection fails closed with a null code");
check(toAppInvokeError("c", undefined).message.includes("c"), "an opaque rejection still names the command");

const already = new AppInvokeError("c", "m", "invalid");
check(toAppInvokeError("other", already) === already, "normalization is idempotent");

// Retry classification. Getting this wrong turns a permanent failure into a
// 3-second hot loop, or gives up on a writer that is mid-publish.
check(isTransientAppError(toAppInvokeError("c", { code: "io", message: "m" })), "io is transient");
check(isTransientAppError(toAppInvokeError("c", { code: "not_found", message: "m" })), "not_found is transient");
check(!isTransientAppError(toAppInvokeError("c", { code: "invalid", message: "m" })), "invalid is permanent");
check(!isTransientAppError(toAppInvokeError("c", { code: "subprocess", message: "m" })), "subprocess is permanent");
check(!isTransientAppError(toAppInvokeError("c", { code: "task", message: "m" })), "task is permanent");
check(!isTransientAppError(toAppInvokeError("c", { code: "unknown_run", message: "m" })), "unknown_run is permanent");
check(!isTransientAppError(new Error("not from IPC")), "a non-IPC error is not treated as transient");
check(!isTransientAppError(toAppInvokeError("c", undefined)), "an uncoded rejection fails closed as permanent");

check(appErrorCode(new Error("plain")) === null, "appErrorCode is null for non-IPC errors");

// Guards the Rust-side match in src-tauri/src/error.rs.
check(APP_ERROR_CODES.length === 6, "every AppError variant has a wire code");
for (const code of APP_ERROR_CODES) {
  check(toAppInvokeError("c", { code, message: "m" }).code === code, `code '${code}' round-trips`);
}

console.log(`app invoke error contract: ${passed} checks passed, ${failures.length} failed`);
if (failures.length) {
  failures.forEach((failure) => console.error(`FAIL ${failure}`));
  process.exit(1);
}
