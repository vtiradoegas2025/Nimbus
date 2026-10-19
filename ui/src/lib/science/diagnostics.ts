import { readPath, type ConfigObject } from "@/lib/config/configSchema";

/*
 * Pre-flight config diagnostics: derived stats (cells, memory, steps, advective CFL = dt·U_max/dx)
 * and validation warnings. CFL uses U_max = 50 m/s as a coarse estimate.
 */

export type DiagTone = "ok" | "info" | "warn" | "error";

export interface DiagStat {
  label: string;
  value: string;
  tone: DiagTone;
  hint?: string;
}

export interface DiagWarning {
  message: string;
  tone: "warn" | "error";
}

export interface ConfigDiagnostics {
  stats: DiagStat[];
  warnings: DiagWarning[];
}

const FIELDS_PER_CELL = 12; // ~12 fp64 prognostic + diagnostic fields
const U_MAX_DEFAULT = 50; // m/s, deep-convection cap for advective CFL

export function computeDiagnostics(config: ConfigObject): ConfigDiagnostics {
  const nx = numAt(config, "grid.nx") ?? 0;
  const ny = numAt(config, "grid.ny") ?? 0;
  const nz = numAt(config, "grid.nz") ?? 0;
  const dx = numAt(config, "grid.dx") ?? 0;
  const dy = numAt(config, "grid.dy") ?? dx;
  const dz = numAt(config, "grid.dz") ?? dx;
  // v1 canonical paths: `time.dt_large` (slow-tendency dt), `time.run_time`, `grid.coordinate_system`.
  const dt = numAt(config, "time.dt_large") ?? 0;
  const durationS = numAt(config, "time.run_time") ?? 0;
  const coord = strAt(config, "grid.coordinate_system") ?? "cartesian";
  const scheme = strAt(config, "dynamics.scheme") ?? "";
  const microScheme = strAt(config, "microphysics.scheme") ?? "none";
  const sfcQv = numAt(config, "initial_conditions.sounding.moisture_qv_kgkg") ?? 0;
  const soundingType = strAt(config, "initial_conditions.sounding.type") ?? "";
  const soundingFile = strAt(config, "initial_conditions.sounding.file_path") ?? "";

  const cells = nx * ny * nz;
  const memBytes = cells * 8 * FIELDS_PER_CELL;
  const memMB = memBytes / (1024 * 1024);
  const steps = dt > 0 ? Math.round(durationS / dt) : 0;
  const minSpacing = Math.max(1, Math.min(dx, dy, dz));
  const cfl = dt > 0 && minSpacing > 0 ? (dt * U_MAX_DEFAULT) / minSpacing : 0;

  const stats: DiagStat[] = [
    {
      label: "cells",
      value: cells > 0 ? cells.toLocaleString() : "—",
      tone: cells > 4_000_000 ? "warn" : "info",
      hint: cells > 4_000_000 ? "Large grid; expect slow per-step wall time." : undefined,
    },
    {
      label: "memory",
      value: memMB > 0 ? formatMem(memMB) : "—",
      tone: memMB > 4096 ? "error" : memMB > 1024 ? "warn" : "info",
      hint:
        memMB > 4096
          ? "Above 4 GB — may not fit in typical laptop RAM."
          : memMB > 1024
            ? "Above 1 GB — close other apps."
            : undefined,
    },
    {
      label: "steps",
      value: steps > 0 ? steps.toLocaleString() : "—",
      tone: steps > 100_000 ? "warn" : "info",
    },
    {
      label: "CFL",
      value: cfl > 0 ? cfl.toFixed(2) : "—",
      tone: cfl > 1 ? "error" : cfl > 0.7 ? "warn" : "ok",
      hint:
        cfl > 1
          ? "Advective CFL > 1 with U=50 m/s. Reduce dt or increase dx."
          : cfl > 0.7
            ? "CFL near the safety margin (0.7)."
            : undefined,
    },
  ];

  const warnings: DiagWarning[] = [];

  // Guard: Cartesian-only; catches uploaded/hand-edited configs with stale coordinate.
  const requiredCoord = requiredCoordinateForScheme(scheme);
  if (requiredCoord && coord !== requiredCoord) {
    warnings.push({
      message: `Scheme '${scheme}' requires coordinate_system: ${requiredCoord} (currently ${coord || "unset"}).`,
      tone: "error",
    });
  }

  // Steer toward C-grid while describing the narrower admitted A-grid families.
  if (scheme === "klemp_wilhelmson_split_cartesian_agrid") {
    warnings.push({
      message:
        "Cartesian + collocated has qualified flat CPU families and an exact " +
        "stationary Bell/Schär mapped CPU family across all retained microphysics, " +
        "chaos, and none/TVD/WENO5 advection. Other geometry, motion, nesting, and " +
        "accelerator combinations remain exact-profile constrained. Use C-grid for " +
        "the broad production profile.",
      tone: "warn",
    });
  }

  // File sounding requires a path.
  if (soundingType === "file" && !soundingFile.trim()) {
    warnings.push({
      message: "Sounding source set to 'file' but no file_path provided.",
      tone: "error",
    });
  }

  // Only parametric_cape uses explicit surface-qv; skip for file/derived soundings.
  if (
    microScheme !== "none" &&
    soundingType === "parametric_cape" &&
    sfcQv <= 0
  ) {
    warnings.push({
      message: `Microphysics scheme '${microScheme}' set with sfc_qv_kgkg=0 — no moisture available for condensation.`,
      tone: "warn",
    });
  }

  // dz vs dx aspect: extreme aspect ratios usually mean a typo.
  if (dx > 0 && dz > 0) {
    const ratio = dz / dx;
    if (ratio > 5 || ratio < 0.2) {
      warnings.push({
        message: `dz/dx aspect ratio is ${ratio.toFixed(2)} — unusual; double-check spacings.`,
        tone: "warn",
      });
    }
  }

  return { stats, warnings };
}

function formatMem(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  if (mb >= 10) return `${Math.round(mb)} MB`;
  return `${mb.toFixed(1)} MB`;
}

/** Coordinate a scheme requires, or null (unknown → stay quiet). Keep in sync with configSchema.ts. */
function requiredCoordinateForScheme(scheme: string): "cartesian" | null {
  if (!scheme) return null;
  if (scheme.includes("cartesian")) return "cartesian";
  switch (scheme) {
    case "cart":
    case "cartesian_cpu":
      return "cartesian";
    default:
      return null;
  }
}

function numAt(config: ConfigObject, path: string): number | undefined {
  const v = readPath(config, path);
  return typeof v === "number" ? v : undefined;
}

function strAt(config: ConfigObject, path: string): string | undefined {
  const v = readPath(config, path);
  return typeof v === "string" ? v : undefined;
}
