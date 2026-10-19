import "./prepare.mjs";
import { register } from "node:module";

// Installed via `node --import ../tests/ui/register-ts-resolve.mjs` before the
// test entry module loads, so extension-less src imports resolve to their .ts
// files (see ts-resolve-hook.mjs).
register("./ts-resolve-hook.mjs", import.meta.url);
