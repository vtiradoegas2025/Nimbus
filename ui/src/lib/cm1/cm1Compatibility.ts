import type {
  Cm1CaseInventory,
  Cm1NamelistAssignment,
  Cm1NamelistReport,
  Cm1SoundingReport,
} from "@/lib/dataset/datasetSources";
import { summarizeCm1Sounding } from "./cm1SoundingSummary";
import { uniqueCm1NamelistValue } from "./cm1NamelistSummary";
import { buildCm1SoundingProjection } from "./cm1SoundingProjection";

export type Cm1CompatibilityStatus =
  | "ready"
  | "needs_decision"
  | "different"
  | "unsupported";

export interface Cm1CompatibilityEntry {
  id: string;
  group: string;
  key: string;
  rawValue: string;
  status: Cm1CompatibilityStatus;
  target: string | null;
  reason: string;
}

export interface Cm1CompatibilityAssetCheck {
  id: string;
  label: string;
  status: Cm1CompatibilityStatus;
  detail: string;
}

export interface Cm1CompatibilityReport {
  schema: "nimbus.studio.cm1-compatibility-report";
  schemaVersion: 1;
  entries: Cm1CompatibilityEntry[];
  assetChecks: Cm1CompatibilityAssetCheck[];
  counts: Record<Cm1CompatibilityStatus, number>;
  draftEligible: false;
  draftBlockers: string[];
}

interface StaticRule {
  status: Cm1CompatibilityStatus;
  target: string | null;
  reason: string;
}

const direct = (target: string, reason: string): StaticRule => ({
  status: "ready",
  target,
  reason,
});

const STATIC_RULES: Readonly<Record<string, StaticRule>> = {
  nx: direct("grid.nx", "Same Cartesian cell-count scalar."),
  ny: direct("grid.ny", "Same Cartesian cell-count scalar."),
  nz: direct("grid.nz", "Same vertical cell-count scalar."),
  dx: direct("grid.dx", "Same metre-valued uniform x spacing."),
  dy: direct("grid.dy", "Same metre-valued uniform y spacing."),
  dz: direct("grid.dz", "Same metre-valued uniform z spacing when CM1 is not stretched."),
  run_time: direct("time.run_time", "Same additional integration duration in seconds."),
  timax: direct("time.run_time", "Same absolute run limit in seconds when run_time is absent."),
  dtl: direct("time.dt_large", "Same large dynamics time step in seconds."),
  nsound: direct("time.acoustic_substeps", "Same explicit acoustic-substep count; zero/automatic behavior still needs preflight."),
  tapfrq: direct("time.output_interval", "Same three-dimensional history cadence in seconds."),
  stretch_z: {
    status: "needs_decision",
    target: "grid.ztop",
    reason: "Nimbus cannot reproduce an arbitrary CM1 stretched-level definition from this flag alone.",
  },
  ztop: {
    status: "needs_decision",
    target: "grid.ztop",
    reason: "The height transfers directly, but the complete CM1 stretching specification must be resolved first.",
  },
  wbc: { status: "needs_decision", target: "boundary_conditions.west", reason: "CM1 numeric boundary modes require an explicit enum translation and numerical-behavior review." },
  ebc: { status: "needs_decision", target: "boundary_conditions.east", reason: "CM1 numeric boundary modes require an explicit enum translation and numerical-behavior review." },
  sbc: { status: "needs_decision", target: "boundary_conditions.south", reason: "CM1 numeric boundary modes require an explicit enum translation and numerical-behavior review." },
  nbc: { status: "needs_decision", target: "boundary_conditions.north", reason: "CM1 numeric boundary modes require an explicit enum translation and numerical-behavior review." },
  bbc: { status: "needs_decision", target: "boundary_conditions.bottom", reason: "Free-slip/no-slip modes have candidates; semi-slip has no direct Nimbus control." },
  tbc: { status: "needs_decision", target: "boundary_conditions.top", reason: "Free-slip/no-slip modes have candidates; CM1 semi-slip coupling has no direct Nimbus control." },
  irdamp: { status: "different", target: "damping.enabled", reason: "Nimbus has an upper sponge, but CM1 modes damp different reference states." },
  zd: { status: "different", target: "damping.z_d", reason: "Same onset-height concept within a scientifically different damping formulation." },
  rdalpha: { status: "different", target: "damping.alpha", reason: "Same inverse-time dimension within a scientifically different damping formulation." },
  icor: { status: "needs_decision", target: "coriolis.enabled", reason: "CM1 mode selection must be reconciled with Nimbus's f-plane configuration." },
  lat0: { status: "needs_decision", target: "coriolis.latitude_deg", reason: "Latitude is usable only when the selected CM1 Coriolis mode derives f from latitude." },
  fcor: { status: "needs_decision", target: "coriolis.latitude_deg", reason: "Nimbus currently exposes latitude rather than a direct Coriolis-parameter scalar." },
  cm1setup: { status: "needs_decision", target: null, reason: "This compound CM1 turbulence/PBL/DNS mode spans several independent Nimbus sections." },
  tconfig: { status: "different", target: "turbulence.mode", reason: "Both distinguish isotropic/anisotropic mixing, but their coefficient formulations are not equivalent." },
  ipbl: { status: "different", target: "boundary_layer.scheme", reason: "Scheme families overlap only partially and require option-specific qualification." },
  difforder: { status: "different", target: "diffusion.scheme", reason: "Nimbus diffusion selection does not reproduce CM1's order switch directly." },
  kdiff2: { status: "different", target: "diffusion.coefficient", reason: "Coefficient meaning depends on the selected operator and cannot be copied independently." },
  kdiff6: { status: "different", target: "diffusion.coefficient", reason: "CM1's sixth-order stability fraction is not Nimbus's coefficient contract." },
  psolver: { status: "different", target: "dynamics.scheme", reason: "Nimbus does not expose CM1's pressure-solver family as a selectable equivalent." },
  isnd: { status: "needs_decision", target: "initial_conditions.sounding.type", reason: "External and analytic CM1 sounding modes require source-specific translation." },
  iwnd: { status: "needs_decision", target: "initial_conditions.hodograph.type", reason: "CM1 wind initialization must be reconciled with the sounding and Nimbus hodograph source." },
  output_format: {
    status: "needs_decision",
    target: "output.format",
    reason: "CM1 output-format codes and layouts do not transfer mechanically. Choose Nimbus netCDF only when exact-binary preflight reports it available.",
  },
  output_filetype: { status: "unsupported", target: null, reason: "Nimbus has no equivalent single-file/per-time/per-rank output layout control." },
  rstfrq: {
    status: "needs_decision",
    target: "extensions.checkpoint.every_s",
    reason:
      "Nimbus now has a seconds-based checkpoint cadence, but its checkpoint format is not a CM1 restart. Review the cadence explicitly; CM1 restart files are never reused.",
  },
  irst: { status: "unsupported", target: null, reason: "Nimbus cannot resume a CM1 restart artifact." },
  rstnum: { status: "unsupported", target: null, reason: "Nimbus cannot select a CM1 restart sequence number." },
  restart_format: { status: "unsupported", target: null, reason: "CM1 restart containers are not Nimbus checkpoint containers." },
  restart_filetype: { status: "unsupported", target: null, reason: "CM1 restart layout has no Nimbus equivalent." },
  axisymm: { status: "unsupported", target: "dynamics.axisymmetric", reason: "Nimbus Studio is Cartesian-only and pins its legacy axisymmetric flag off." },
  imove: { status: "unsupported", target: null, reason: "A translating CM1 domain has no Nimbus runtime equivalent." },
  umove: { status: "unsupported", target: null, reason: "Storm-relative domain translation is not implemented." },
  vmove: { status: "unsupported", target: null, reason: "Storm-relative domain translation is not implemented." },
};

function numeric(raw: string): number | null {
  const normalized = raw.trim().replace(/[dD]/g, "e");
  if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(normalized)) return null;
  const value = Number(normalized);
  return Number.isFinite(value) ? value : null;
}

function logical(raw: string | undefined): boolean | null {
  if (raw == null) return null;
  const value = raw.trim().toLowerCase();
  if (value === ".true." || value === "true" || value === "t") return true;
  if (value === ".false." || value === "false" || value === "f") return false;
  return null;
}

function dynamicRule(key: string, rawValue: string): StaticRule | null {
  const value = numeric(rawValue);
  if (key === "ptype") {
    if (value === 0) return direct("microphysics.scheme", "Microphysics disabled maps to Nimbus none.");
    if (value === 1) return { status: "different", target: "microphysics.scheme=kessler", reason: "Both are Kessler-family warm rain, but implementation identity is not established." };
    if (value === 3) return { status: "different", target: "microphysics.scheme=thompson", reason: "Nimbus's Thompson-labeled scheme is not CM1's Thompson implementation." };
    if (value === 5) return { status: "different", target: "microphysics.scheme=milbrandt", reason: "Nimbus Milbrandt-Yau and CM1 Morrison are distinct double-moment schemes; no alias or implementation equivalence is claimed." };
    return { status: "unsupported", target: null, reason: `CM1 ptype=${rawValue.trim()} has no qualified Nimbus implementation mapping.` };
  }
  if (key === "sgsmodel") {
    if (value === 1) return { status: "different", target: "turbulence.scheme=tke", reason: "Both are TKE-family closures, but the implementations and tunables are not equivalent." };
    if (value === 2) return { status: "different", target: "turbulence.scheme=smagorinsky", reason: "Both are Smagorinsky-family closures, but the implementations and tunables are not equivalent." };
    return { status: "unsupported", target: null, reason: `CM1 sgsmodel=${rawValue.trim()} has no qualified Nimbus closure.` };
  }
  if (key.startsWith("output_")) {
    return { status: "needs_decision", target: "output.fields", reason: "CM1 per-variable output flags must be reconciled with Nimbus field bundles and derived diagnostics." };
  }
  return null;
}

function classifyAssignment(
  group: string,
  assignment: Cm1NamelistAssignment,
  occurrence: number,
): Cm1CompatibilityEntry {
  const key = assignment.key.toLowerCase();
  const indexed = key.includes("(");
  const rule = indexed
    ? null
    : STATIC_RULES[key] ?? dynamicRule(key, assignment.rawValue) ?? null;
  const numericReady = rule?.status === "ready" && key !== "ptype";
  return {
    id: `${group}.${key}#${occurrence}`,
    group,
    key,
    rawValue: assignment.rawValue,
    status: numericReady && numeric(assignment.rawValue) == null
      ? "needs_decision"
      : rule?.status ?? "unsupported",
    target: rule?.target ?? null,
    reason: numericReady && numeric(assignment.rawValue) == null
      ? "The target is known, but this is not one unambiguous numeric scalar and cannot transfer mechanically."
      : rule?.reason ?? "No maintained CM1-to-Nimbus mapping rule exists; the value will not enter a draft.",
  };
}

function assetChecks(
  inventory: Cm1CaseInventory | undefined,
  namelist: Cm1NamelistReport,
  sounding: Cm1SoundingReport | undefined,
  namelistInspectionError?: string,
  soundingInspectionError?: string,
): Cm1CompatibilityAssetCheck[] {
  const checks: Cm1CompatibilityAssetCheck[] = [];
  checks.push({
    id: "namelist",
    label: "namelist.input",
    status: namelistInspectionError || namelist.warnings.length > 0 ? "needs_decision" : "ready",
    detail: namelistInspectionError
      ? `Inspection failed: ${namelistInspectionError}`
      : namelist.warnings.length > 0
        ? `${namelist.warnings.length} parser warning${namelist.warnings.length === 1 ? "" : "s"} must be reviewed.`
        : `${namelist.assignmentCount} assignments retained in source order.`,
  });
  if (inventory) {
    checks.push({
      id: "history",
      label: "CM1 history",
      status: inventory.historyFiles.length > 0 ? "ready" : "needs_decision",
      detail: inventory.historyFiles.length > 0
        ? `${inventory.historyFiles.length} top-level netCDF history file${inventory.historyFiles.length === 1 ? "" : "s"} available for qualified analysis.`
        : "No history is present; this case is available for setup inspection only.",
    });
    if (inventory.restartFiles.length > 0) {
      checks.push({
        id: "restarts",
        label: "CM1 restarts",
        status: "unsupported",
        detail: `${inventory.restartFiles.length} restart candidate${inventory.restartFiles.length === 1 ? "" : "s"} preserved, but Nimbus cannot resume them.`,
      });
    }
    if (inventory.statisticsFiles.length > 0) {
      checks.push({
        id: "statistics",
        label: "CM1 statistics",
        status: "unsupported",
        detail: "cm1out_stats.dat is preserved but does not yet have a Studio reader.",
      });
    }
  }
  const isnd = uniqueCm1NamelistValue(namelist, "isnd");
  const expectsExternalSounding = numeric(isnd ?? "") === 7 || numeric(isnd ?? "") === 17;
  if (soundingInspectionError) {
    checks.push({
      id: "sounding",
      label: "input_sounding",
      status: "needs_decision",
      detail: `Inspection failed: ${soundingInspectionError}`,
    });
  } else if (sounding) {
    const summary = summarizeCm1Sounding(sounding, namelist);
    const projection = buildCm1SoundingProjection(sounding, namelist);
    checks.push({
      id: "sounding",
      label: "input_sounding",
      status: !projection.copyEligible || summary.attention
        ? "needs_decision"
        : "different",
      detail: summary.attention
        ?? (projection.copyEligible
          ? `${summary.facts[0]} retained; a reviewable hydrostatic Nimbus text projection is available, but exact initialization equivalence is unqualified.`
          : projection.blockers[0] ?? "The source profile cannot be projected safely."),
    });
  } else if (expectsExternalSounding) {
    checks.push({
      id: "sounding",
      label: "input_sounding",
      status: "unsupported",
      detail: `CM1 isnd=${isnd?.trim()} requires input_sounding, but no readable profile was discovered.`,
    });
  }
  return checks;
}

export function buildCm1CompatibilityReport(args: {
  namelist: Cm1NamelistReport;
  inventory?: Cm1CaseInventory;
  sounding?: Cm1SoundingReport;
  namelistInspectionError?: string;
  soundingInspectionError?: string;
}): Cm1CompatibilityReport {
  const seen = new Map<string, number>();
  const initialEntries = args.namelist.groups.flatMap((group) =>
    group.assignments.map((assignment) => {
      const identity = `${group.name}.${assignment.key.toLowerCase()}`;
      const occurrence = (seen.get(identity) ?? 0) + 1;
      seen.set(identity, occurrence);
      return classifyAssignment(group.name, assignment, occurrence);
    }),
  );
  const identityCounts = new Map<string, number>();
  for (const entry of initialEntries) {
    const identity = `${entry.group}.${entry.key}`;
    identityCounts.set(identity, (identityCounts.get(identity) ?? 0) + 1);
  }
  const hasRunTime = initialEntries.some((entry) => entry.key === "run_time");
  const stretched = logical(uniqueCm1NamelistValue(args.namelist, "stretch_z") ?? undefined);
  const entries = initialEntries.map((entry): Cm1CompatibilityEntry => {
    if ((identityCounts.get(`${entry.group}.${entry.key}`) ?? 0) > 1) {
      return {
        ...entry,
        status: "needs_decision",
        reason: "This source identity is assigned more than once; precedence must be reviewed before translation.",
      };
    }
    if (entry.key === "timax" && hasRunTime) {
      return {
        ...entry,
        status: "needs_decision",
        reason: "CM1 run_time overrides timax; this shadowed value cannot also populate Nimbus time.run_time.",
      };
    }
    if (entry.key === "dz" && stretched === true) {
      return {
        ...entry,
        status: "needs_decision",
        reason: "CM1 vertical stretching is enabled, so one dz scalar does not describe the translated vertical grid.",
      };
    }
    return entry;
  });
  const checks = assetChecks(
    args.inventory,
    args.namelist,
    args.sounding,
    args.namelistInspectionError,
    args.soundingInspectionError,
  );
  const counts: Record<Cm1CompatibilityStatus, number> = {
    ready: 0,
    needs_decision: 0,
    different: 0,
    unsupported: 0,
  };
  for (const item of [...entries, ...checks]) counts[item.status] += 1;
  const draftBlockers = [
    counts.needs_decision > 0 ? `${counts.needs_decision} item${counts.needs_decision === 1 ? " needs" : "s need"} a decision` : null,
    counts.different > 0 ? `${counts.different} scientifically different item${counts.different === 1 ? "" : "s"}` : null,
    counts.unsupported > 0 ? `${counts.unsupported} unsupported or unmapped item${counts.unsupported === 1 ? "" : "s"}` : null,
    "Runnable Nimbus configuration generation is not enabled for compatibility reports",
  ].filter((value): value is string => value != null);
  return {
    schema: "nimbus.studio.cm1-compatibility-report",
    schemaVersion: 1,
    entries,
    assetChecks: checks,
    counts,
    draftEligible: false,
    draftBlockers,
  };
}
