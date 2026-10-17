import { useEffect, useMemo, useState } from "react";
import type { Annotations, Data, Layout } from "plotly.js";
import { profile1D } from "@/lib/platform/npy";
import {
  celsiusToKelvin,
  columnDewpointCelsius,
  columnTemperatureCelsius,
  detectPressureUnit,
  dryAdiabat,
  moistAdiabat,
  P_REF_HPA,
  parcelAscent,
  parcelDiagnostics,
  pressureUnitFromLabel,
  pToY,
  saturationMixingRatioLine,
  SKEW_C_PER_DECADE,
  skewX,
  toHPa,
  type PressureUnit,
} from "@/lib/science/thermo";
import { Icon } from "@/components/shell/Icon";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import {
  ControlGroup,
  SCRUB_DEBOUNCE_MS,
  useDebouncedFrameIdentity,
  useField,
  usePlotChrome,
  useRetainedPlot,
  type PlotChrome,
  type PlotComponentProps,
} from "./common";
import { axisLabel } from "@/lib/science/gridCoords";
import { findPublishedFrame, loadedFramesAreCoherent } from "@/lib/run/frameIdentity";
import { cropToValidPlotDomain, validAxisLength } from "@/lib/science/validDomain";
import type { Npy3D } from "@/lib/platform/npy";
import { unitFor } from "@/lib/science/fieldUnits";
import { planExactLineWindows, windowsForFieldAggregation } from "@/lib/plot/fieldLoadPlanning";
import { useOutputHierarchySource } from "@/lib/run/useOutputHierarchySource";
import { OutputHierarchyControls } from "./OutputHierarchyControls";
import { PlotCompressionControls } from "./PlotSourceDisclosure";
import { fieldsForPlotRanks } from "@/lib/plot/fieldPlotEligibility";

/* Skew-T log-P: display transform applied to column profiles (p, theta/temperature, qv). */

const Y_TICK_PHPA = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100];
const ISOTHERMS_C = [
  -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40,
];
const DRY_ADIABATS_K = [
  220, 240, 260, 280, 300, 310, 320, 330, 340, 360, 380, 400,
];
// Moist (pseudo) adiabats labeled by their 1000 hPa temperature (≈ θ_w).
const MOIST_ADIABATS_T1000_C = [-20, -10, 0, 8, 16, 24, 32];
// Saturation mixing-ratio isohumes (g/kg) — the conventional skew-T set.
const MIXING_RATIO_GKG = [0.4, 1, 2, 4, 7, 10, 16, 24, 32];
/* Minimum skewed-x span to keep isotherms legible for thin/near-isothermal columns. */
const MIN_SKEWED_X_SPAN = 40;

const BG_ISOTHERM_COLOR = "rgba(91, 155, 255, 0.18)";
const BG_ADIABAT_COLOR = "rgba(217, 164, 65, 0.18)";
const BG_MOIST_ADIABAT_COLOR = "rgba(93, 191, 129, 0.26)";
const BG_MIXING_RATIO_COLOR = "rgba(180, 130, 220, 0.24)";
const T_COLOR = "#e26370";
const TD_COLOR = "#5dbf81";
// Parcel trace uses chrome.ink (theme-neutral) so it inverts correctly on both themes.

export function SkewT({
  runDir: rootRunDir,
  steps: rootSteps,
  stepIdx: rootStepIdx,
}: PlotComponentProps) {
  const rootCurrentStep = rootSteps[rootStepIdx] ?? null;
  const rootFields = useMemo(() => rootCurrentStep?.fields ?? [], [rootCurrentStep?.fields]);
  const rootTemperature = rootFields.includes("temperature")
    ? "temperature"
    : rootFields.includes("theta") ? "theta" : null;
  const hierarchyRequiredFields = useMemo(
    () => [
      rootTemperature,
      rootFields.includes("p") ? "p" : null,
      rootFields.includes("qv") ? "qv" : null,
    ].filter((field): field is string => field != null),
    [rootFields, rootTemperature],
  );
  const outputHierarchySource = useOutputHierarchySource({
    runDir: rootRunDir,
    steps: rootSteps,
    stepIdx: rootStepIdx,
    scope: "shared",
    requiredFields: hierarchyRequiredFields,
  });
  const runDir = outputHierarchySource.sourceRunDir;
  const steps = outputHierarchySource.sourceSteps;
  const currentStep = outputHierarchySource.currentStep;
  const volumeFields = fieldsForPlotRanks(runDir, currentStep, [3]);
  const grid = currentStep?.grid ?? null;
  // Index-picker axis letters for the Cartesian grid: x/y/z.
  const axis0Label = axisLabel(0, grid);
  const axis1Label = axisLabel(1, grid);
  const hasTemperature = volumeFields.includes("temperature");
  const hasTheta = volumeFields.includes("theta");
  const hasP = volumeFields.includes("p");
  const hasQv = volumeFields.includes("qv");

  const tempFieldName = hasTemperature ? "temperature" : hasTheta ? "theta" : null;

  const [rIdx, setRIdx] = useState(0);
  const [thIdx, setThIdx] = useState(0);
  const chrome = usePlotChrome();

  // Debounce the exact sounding-column reads so a scrub/autoplay loads only the
  // settled frame. Recording bypasses the debounce.
  const loadFrame = useDebouncedFrameIdentity(currentStep, SCRUB_DEBOUNCE_MS);
  const linePlan = useMemo(() => {
    const outcome = planExactLineWindows(grid, 2, [rIdx, thIdx, 0]);
    return outcome.ok ? outcome.plan.windows : null;
  }, [grid, rIdx, thIdx]);
  const t = useField(
    runDir, linePlan && tempFieldName ? loadFrame.stepIndex : null,
    tempFieldName, loadFrame.publicationGeneration,
    linePlan && tempFieldName ? windowsForFieldAggregation(linePlan, tempFieldName) : null,
  );
  const p = useField(
    runDir, linePlan && hasP ? loadFrame.stepIndex : null,
    hasP ? "p" : null, loadFrame.publicationGeneration,
    linePlan ? windowsForFieldAggregation(linePlan, "p") : null,
  );
  const qv = useField(
    runDir, linePlan && hasQv ? loadFrame.stepIndex : null,
    hasQv ? "qv" : null, loadFrame.publicationGeneration,
    linePlan ? windowsForFieldAggregation(linePlan, "qv") : null,
  );
  const requiredLoads = hasQv ? [t, p, qv] : [t, p];
  const coherentFrame = loadedFramesAreCoherent(...requiredLoads);
  const loadedStep = coherentFrame
    ? findPublishedFrame(steps, t.loadedStepIndex, t.loadedPublicationGeneration)
    : null;
  const tField = useMemo(
    () => coherentFrame && t.field
      ? cropToValidPlotDomain(t.field, loadedStep?.grid).field as Npy3D
      : null,
    [coherentFrame, t.field, loadedStep?.grid],
  );
  const pField = useMemo(
    () => coherentFrame && p.field
      ? cropToValidPlotDomain(p.field, loadedStep?.grid).field as Npy3D
      : null,
    [coherentFrame, p.field, loadedStep?.grid],
  );
  const qvField = useMemo(
    () => coherentFrame && qv.field
      ? cropToValidPlotDomain(qv.field, loadedStep?.grid).field as Npy3D
      : null,
    [coherentFrame, qv.field, loadedStep?.grid],
  );
  const catalogPressureLabel = unitFor("p", loadedStep?.fieldUnits);
  const catalogPressureUnit = pressureUnitFromLabel(catalogPressureLabel);
  const unsupportedPressureUnit = catalogPressureLabel != null && catalogPressureUnit == null
    ? catalogPressureLabel
    : null;

  const validNr = validAxisLength(0, grid) ?? grid?.nr ?? 0;
  const validNth = validAxisLength(1, grid) ?? grid?.nth ?? 0;
  useEffect(() => {
    if (validNr > 0) setRIdx((prev) => Math.min(prev, validNr - 1));
    if (validNth > 0) setThIdx((prev) => Math.min(prev, validNth - 1));
  }, [validNr, validNth]);

  const plot = useMemo(() => {
    if (!tField || !pField) return null;
    if (
      tField.shape[0] !== pField.shape[0] ||
      tField.shape[1] !== pField.shape[1] ||
      tField.shape[2] !== pField.shape[2]
    ) {
      return null;
    }
    if (qvField) {
      if (
        qvField.shape[0] !== pField.shape[0] ||
        qvField.shape[1] !== pField.shape[1] ||
        qvField.shape[2] !== pField.shape[2]
      ) {
        return null;
      }
    }

    const pProfile = profile1D(pField, 2, 0, 0);
    const tProfile = profile1D(tField, 2, 0, 0);
    const qvProfile = qvField ? profile1D(qvField, 2, 0, 0) : null;

    if (unsupportedPressureUnit) return null;
    // Current publications carry the field contract's exact unit. Magnitude
    // inspection is retained only for old manifests without a unit catalog.
    const pUnit: PressureUnit = catalogPressureUnit ?? detectPressureUnit(pProfile);
    const pHPa = new Float64Array(pProfile.length);
    const pPaForCalc = new Float64Array(pProfile.length);
    for (let k = 0; k < pProfile.length; k++) {
      pHPa[k] = toHPa(pProfile[k], pUnit);
      pPaForCalc[k] = pUnit === "Pa" ? pProfile[k] : pProfile[k] * 100;
    }

    const tC = columnTemperatureCelsius(tProfile, pPaForCalc, hasTemperature);
    const tdC = qvProfile ? columnDewpointCelsius(qvProfile, pPaForCalc) : null;

    // Lifted surface parcel + CAPE/CIN, when moisture (qv) is available.
    const parcel = qvProfile ? computeParcelOverlay(tC, qvProfile, pHPa) : null;

    return buildPlot(tC, tdC, pHPa, pUnit, rIdx, thIdx, axis0Label, axis1Label, chrome, parcel);
  }, [
    tField,
    pField,
    qvField,
    hasTemperature,
    rIdx,
    thIdx,
    axis0Label,
    axis1Label,
    catalogPressureUnit,
    unsupportedPressureUnit,
    chrome,
  ]);

  // Three fields resolve independently, so the coherence gate above nulls the figure on every frame swap.
  const { shown: shownPlot } = useRetainedPlot(
    plot,
    `${runDir}|${tempFieldName ?? ""}|${loadFrame.publicationGeneration ?? ""}`,
  );

  const loading = t.loading || p.loading || qv.loading;
  const error = t.error ?? p.error ?? qv.error;

  const missing: string[] = [];
  if (!tempFieldName) missing.push("temperature or theta");
  if (!hasP) missing.push("p");
  // qv is optional — drives the dewpoint line; absence just hides T_d.

  const maxR = Math.max(1, validNr);
  const maxTh = Math.max(1, validNth);
  const usingTheta = !hasTemperature && hasTheta;
  // Rendered in a fixed-height toolbar slot that ellipsizes, so the full
  // sentence also goes on the title for narrow windows.
  const sourceSummary = `T from ${usingTheta ? "θ" : "temperature"}${hasQv ? "" : ", no Td (qv missing)"}`;

  return (
    <>
      <div className="plot-controls">
        <OutputHierarchyControls source={outputHierarchySource} />
        <PlotCompressionControls step={loadedStep} fields={[tempFieldName, "p", hasQv ? "qv" : null]} />
        <ControlGroup label={`${axis0Label} index (${rIdx} / ${maxR - 1})`} stretch>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxR - 1)}
            value={rIdx}
            onChange={(e) => setRIdx(Number(e.target.value))}
            disabled={!tField || !pField}
          />
        </ControlGroup>

        <ControlGroup label={`${axis1Label} index (${thIdx} / ${maxTh - 1})`} stretch>
          <input
            type="range"
            min={0}
            max={Math.max(0, maxTh - 1)}
            value={thIdx}
            onChange={(e) => setThIdx(Number(e.target.value))}
            disabled={!tField || !pField}
          />
        </ControlGroup>

        <ControlGroup label="Source">
          <span className="plot-controls__status" title={sourceSummary}>
            {sourceSummary}
          </span>
        </ControlGroup>
      </div>

      <div className="plot-body">
        {missing.length > 0 && (
          <div className="plot-overlay plot-overlay--warn">
            <Icon name="warning" size={12} />
            <span>Missing required field(s): {missing.join(", ")}</span>
          </div>
        )}
        {loading && (
          <div className="plot-overlay">
            <Icon name="dot" size={12} />
            <span>Loading…</span>
          </div>
        )}
        {error && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>{error}</span>
          </div>
        )}
        {unsupportedPressureUnit && (
          <div className="plot-overlay plot-overlay--error">
            <Icon name="warning" size={12} />
            <span>Unsupported pressure unit in manifest: {unsupportedPressureUnit}</span>
          </div>
        )}
        {shownPlot ? (
          <PlotCanvas data={shownPlot.data} layout={shownPlot.layout} />
        ) : (
          !loading &&
          !error &&
          !unsupportedPressureUnit &&
          missing.length === 0 && (
            <div className="plot-overlay">
              <span>No data to plot.</span>
            </div>
          )
        )}
      </div>
    </>
  );
}

interface BuiltPlot {
  data: Data[];
  layout: Partial<Layout>;
}

function buildPlot(
  tC: Float32Array,
  tdC: Float32Array | null,
  pHPa: Float64Array | number[] | ArrayLike<number>,
  pUnit: PressureUnit,
  rIdx: number,
  thIdx: number,
  axis0Label: string,
  axis1Label: string,
  chrome: PlotChrome,
  parcel: ParcelOverlay | null = null,
): BuiltPlot {
  // Y range from actual column with log-p margin; 1000/100 hPa only as fallback for degenerate columns.
  let pMax = -Infinity;
  let pMin = Infinity;
  let nFinite = 0;
  for (let i = 0; i < pHPa.length; i++) {
    const v = pHPa[i];
    if (!Number.isFinite(v) || v <= 0) continue;
    nFinite++;
    if (v > pMax) pMax = v;
    if (v < pMin) pMin = v;
  }
  let pBottomHPa: number;
  let pTopHPa: number;
  if (nFinite >= 2 && pMax > pMin) {
    const padLog = (Math.log(pMax) - Math.log(pMin)) * 0.04;
    pBottomHPa = Math.exp(Math.log(pMax) + padLog);
    pTopHPa = Math.exp(Math.log(pMin) - padLog);
  } else {
    pBottomHPa = 1000;
    pTopHPa = 100;
  }

  const yBottom = pToY(pBottomHPa);
  const yTop = pToY(pTopHPa);

  const traces: Data[] = [];

  // Background isotherms (constant T) — straight diagonal lines. The first is
  // a legend proxy so the line family is identifiable; the rest stay hidden.
  ISOTHERMS_C.forEach((t0, idx) => {
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [skewX(t0, pBottomHPa), skewX(t0, pTopHPa)],
      y: [yBottom, yTop],
      line: { color: BG_ISOTHERM_COLOR, width: 1 },
      hoverinfo: "skip",
      showlegend: idx === 0,
      legendgroup: "isotherm",
      name: "isotherm",
    });
  });

  // Background dry adiabats (constant θ) — curve up-left.
  DRY_ADIABATS_K.forEach((thetaK, idx) => {
    const curve = dryAdiabat(thetaK, pTopHPa, pBottomHPa, 40);
    const xs = curve.tC.map((t, i) => skewX(t, curve.pHPa[i]));
    const ys = curve.pHPa.map((p) => pToY(p));
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: BG_ADIABAT_COLOR, width: 1, dash: "dot" },
      hoverinfo: "skip",
      showlegend: idx === 0,
      legendgroup: "dry-adiabat",
      name: "dry adiabat",
    });
  });

  // Background moist (pseudo) adiabats — saturated parcels; asymptote to the
  // dry adiabats aloft as saturation mixing ratio vanishes.
  MOIST_ADIABATS_T1000_C.forEach((t1000, idx) => {
    const curve = moistAdiabat(t1000, pTopHPa, pBottomHPa, 48);
    const xs = curve.tC.map((t, i) => skewX(t, curve.pHPa[i]));
    const ys = curve.pHPa.map((p) => pToY(p));
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: BG_MOIST_ADIABAT_COLOR, width: 1, dash: "dash" },
      hoverinfo: "skip",
      showlegend: idx === 0,
      legendgroup: "moist-adiabat",
      name: "moist adiabat",
    });
  });

  // Background saturation mixing-ratio lines (isohumes), in g/kg.
  MIXING_RATIO_GKG.forEach((w, idx) => {
    const curve = saturationMixingRatioLine(w, pTopHPa, pBottomHPa, 24);
    const xs = curve.tC.map((t, i) => skewX(t, curve.pHPa[i]));
    const ys = curve.pHPa.map((p) => pToY(p));
    traces.push({
      type: "scatter",
      mode: "lines",
      x: xs,
      y: ys,
      line: { color: BG_MIXING_RATIO_COLOR, width: 1, dash: "dashdot" },
      hoverinfo: "skip",
      showlegend: idx === 0,
      legendgroup: "mixing-ratio",
      name: "mixing ratio (g/kg)",
    });
  });

  // Pressure-axis reference horizontals (subtle gridlines at standard
  // levels — Plotly's grid is unaware of the skew transform).
  for (const pTick of Y_TICK_PHPA) {
    if (pTick > pBottomHPa || pTick < pTopHPa) continue;
    const y = pToY(pTick);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: [skewX(-100, pTick), skewX(40, pTick)],
      y: [y, y],
      line: { color: chrome.grid, width: 1 },
      hoverinfo: "skip",
      showlegend: false,
      name: `${pTick} hPa`,
    });
  }

  // x axis carries skewed (not raw temperature) coordinates; fit to the plotted series to avoid clipping.
  let xMin = Infinity;
  let xMax = -Infinity;
  const noteSkewedX = (values: ArrayLike<number>): void => {
    for (let i = 0; i < values.length; i++) {
      const value = values[i];
      if (!Number.isFinite(value)) continue;
      if (value < xMin) xMin = value;
      if (value > xMax) xMax = value;
    }
  };

  // Environment T(p).
  const tX = new Array(tC.length);
  const tY = new Array(tC.length);
  for (let k = 0; k < tC.length; k++) {
    tX[k] = skewX(tC[k], pHPa[k]);
    tY[k] = pToY(pHPa[k]);
  }
  noteSkewedX(tX);
  traces.push({
    type: "scatter",
    mode: "lines+markers",
    x: tX,
    y: tY,
    line: { color: T_COLOR, width: 2 },
    marker: { color: T_COLOR, size: 4 },
    name: "T",
    hovertemplate: `T=%{customdata:.1f}°C<br>p=%{text} hPa<extra></extra>`,
    customdata: Array.from(tC),
    text: Array.from(pHPa, (p) => p.toFixed(0)),
  });

  // Environment T_d(p), if available.
  if (tdC) {
    const tdX = new Array(tdC.length);
    const tdY = new Array(tdC.length);
    for (let k = 0; k < tdC.length; k++) {
      tdX[k] = Number.isFinite(tdC[k]) ? skewX(tdC[k], pHPa[k]) : NaN;
      tdY[k] = pToY(pHPa[k]);
    }
    noteSkewedX(tdX);
    traces.push({
      type: "scatter",
      mode: "lines+markers",
      x: tdX,
      y: tdY,
      line: { color: TD_COLOR, width: 2 },
      marker: { color: TD_COLOR, size: 4 },
      name: "Td",
      hovertemplate: `Td=%{customdata:.1f}°C<br>p=%{text} hPa<extra></extra>`,
      customdata: Array.from(tdC),
      text: Array.from(pHPa, (p) => p.toFixed(0)),
    });
  }

  // Lifted-parcel path + LCL/LFC/EL markers (when a parcel was computed).
  if (parcel && parcel.pHPa.length > 1) {
    const px = parcel.pHPa.map((p, i) => skewX(parcel.tC[i], p));
    const py = parcel.pHPa.map((p) => pToY(p));
    noteSkewedX(px);
    traces.push({
      type: "scatter",
      mode: "lines",
      x: px,
      y: py,
      line: { color: chrome.ink, width: 2.5, dash: "dash" },
      name: "parcel",
      legendgroup: "parcel",
      hovertemplate: `parcel=%{customdata:.1f}°C<br>p=%{text} hPa<extra></extra>`,
      customdata: parcel.tC,
      text: parcel.pHPa.map((p) => p.toFixed(0)),
    });

    // Small horizontal markers at the LCL / LFC / EL, placed on the parcel
    // trace by interpolating the parcel temperature to each pressure.
    const tcAtP = makeLogPInterp(parcel.pHPa, parcel.tC);
    const levels: { label: string; p: number | null }[] = [
      { label: "LCL", p: parcel.lclHPa },
      { label: "LFC", p: parcel.lfcHPa },
      { label: "EL", p: parcel.elHPa },
    ];
    const mx: number[] = [];
    const my: number[] = [];
    const mtext: string[] = [];
    const mp: string[] = [];
    for (const lvl of levels) {
      if (lvl.p == null || !Number.isFinite(lvl.p)) continue;
      const tcAt = tcAtP(lvl.p);
      if (!Number.isFinite(tcAt)) continue;
      mx.push(skewX(tcAt, lvl.p));
      my.push(pToY(lvl.p));
      mtext.push(lvl.label);
      mp.push(lvl.p.toFixed(0));
    }
    if (mx.length > 0) {
      traces.push({
        type: "scatter",
        mode: "text+markers",
        x: mx,
        y: my,
        text: mtext,
        textposition: "middle right",
        textfont: { color: chrome.ink, size: 10 },
        marker: {
          color: chrome.ink,
          size: 12,
          symbol: "line-ew",
          line: { color: chrome.ink, width: 2.5 },
        },
        name: "parcel levels",
        legendgroup: "parcel",
        showlegend: false,
        customdata: mp,
        hovertemplate: `%{text} @ %{customdata} hPa<extra></extra>`,
      });
    }
  }

  // Pressure tick labels (custom because the y values are log10 ratios).
  const yTickVals = Y_TICK_PHPA.filter((p) => p <= pBottomHPa && p >= pTopHPa).map(
    (p) => pToY(p),
  );
  const yTickText = Y_TICK_PHPA.filter((p) => p <= pBottomHPa && p >= pTopHPa).map(
    (p) => `${p}`,
  );

  // Pad the fitted span; floor keeps skew-T proportions for thin/near-isothermal columns.
  let xRange: [number, number] = [-40, 40];
  if (xMax > xMin) {
    const pad = (xMax - xMin) * 0.06;
    let low = xMin - pad;
    let high = xMax + pad;
    const deficit = MIN_SKEWED_X_SPAN - (high - low);
    if (deficit > 0) {
      low -= deficit / 2;
      high += deficit / 2;
    }
    xRange = [low, high];
  }

  const layout: Partial<Layout> = {
    xaxis: {
      title: { text: "T (°C), skewed" },
      range: xRange,
      zeroline: false,
      gridcolor: chrome.grid,
    },
    yaxis: {
      title: { text: `p (hPa, log) — input ${pUnit}` },
      range: [yBottom, yTop],
      tickmode: "array",
      tickvals: yTickVals,
      ticktext: yTickText,
      zeroline: false,
      gridcolor: chrome.grid,
    },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: chrome.overlayBackground },
    title: {
      text: `Skew-T @ (${axis0Label}=${rIdx}, ${axis1Label}=${thIdx})  skew ${SKEW_C_PER_DECADE}°C/dec  ref ${P_REF_HPA} hPa`,
      font: { size: 13 },
    },
    hovermode: "closest",
  };

  // CAPE/CIN readout: CIN is negated here (parcelDiagnostics returns positive inhibition).
  if (parcel) {
    const annotation: Partial<Annotations> = {
      xref: "paper",
      yref: "paper",
      x: 0.98,
      y: 0.98,
      xanchor: "right",
      yanchor: "top",
      align: "right",
      showarrow: false,
      // "dominant layer" is not decoration: this is the largest single positive-buoyancy layer, not
      // the sum of every one, so a sounding with elevated instability reads lower than SBCAPE would.
      text: `CAPE ${Math.round(parcel.capeJkg)} J/kg · dominant layer<br>CIN ${-Math.round(parcel.cinJkg)} J/kg`,
      font: { color: chrome.ink, size: 12 },
      bgcolor: chrome.overlayBackground,
      bordercolor: chrome.ink,
      borderwidth: 1,
      borderpad: 4,
    };
    layout.annotations = [annotation];
  }

  return { data: traces, layout };
}

/** Lifted-parcel overlay derived from the column for the Skew-T. */
interface ParcelOverlay {
  pHPa: number[];
  tC: number[];
  lclHPa: number;
  lfcHPa: number | null;
  elHPa: number | null;
  capeJkg: number;
  cinJkg: number;
}

/** Lift the surface parcel and compute CAPE/CIN. Skipped unless qv is valid at every level. */
function computeParcelOverlay(
  tC: ArrayLike<number>,
  qvProfile: ArrayLike<number>,
  pHPa: ArrayLike<number>,
): ParcelOverlay | null {
  const n = Math.min(tC.length, qvProfile.length, pHPa.length);
  const levels: { p: number; tK: number; qv: number }[] = [];
  for (let k = 0; k < n; k++) {
    const p = pHPa[k];
    const tc = tC[k];
    if (Number.isFinite(p) && Number.isFinite(tc) && p > 0) {
      const qv = qvProfile[k];
      levels.push({ p, tK: celsiusToKelvin(tc), qv: Number.isFinite(qv) ? qv : NaN });
    }
  }
  if (levels.length < 2) return null;
  levels.sort((a, b) => b.p - a.p); // surface (max pressure) first

  // CAPE/CIN uses environmental virtual temperature throughout the column.
  // Replacing a missing qv sample with zero fabricates a dry layer; borrowing
  // the first moist value aloft for the surface fabricates the parcel itself.
  // Either can materially move the LFC/EL and integrated energy, so omit the
  // diagnostic when the moisture column is incomplete.
  if (levels.some((level) => !Number.isFinite(level.qv) || level.qv < 0)) return null;

  const surface = levels[0];
  const qvSfc = surface.qv;

  const pTop = levels[levels.length - 1].p;
  const path = parcelAscent(surface.tK, surface.p, qvSfc, pTop);
  if (path.p_hPa.length < 2) return null;

  const ps = levels.map((l) => l.p);
  const envTempK = makeLogPInterp(ps, levels.map((l) => l.tK));
  const envQv = makeLogPInterp(ps, levels.map((l) => l.qv));
  const diag = parcelDiagnostics(path, envTempK, envQv);

  return {
    pHPa: path.p_hPa,
    tC: path.T_C,
    lclHPa: path.LCL_hPa,
    lfcHPa: diag.LFC_hPa,
    elHPa: diag.EL_hPa,
    capeJkg: diag.CAPE_J_per_kg,
    cinJkg: diag.CIN_J_per_kg,
  };
}

/**
 * Build a piecewise-linear-in-log-p interpolator over descending-pressure
 * samples. Out-of-range pressures clamp to the nearest endpoint value.
 */
function makeLogPInterp(
  psDesc: number[],
  vals: ArrayLike<number>,
): (p: number) => number {
  const last = psDesc.length - 1;
  return (p: number) => {
    if (!(p > 0) || last < 0) return NaN;
    if (p >= psDesc[0]) return vals[0];
    if (p <= psDesc[last]) return vals[last];
    const x = Math.log(p);
    for (let i = 0; i < last; i++) {
      const pHi = psDesc[i];
      const pLo = psDesc[i + 1];
      if (p <= pHi && p >= pLo) {
        const xHi = Math.log(pHi);
        const xLo = Math.log(pLo);
        const f = (x - xHi) / (xLo - xHi);
        return vals[i] + f * (vals[i + 1] - vals[i]);
      }
    }
    return vals[last];
  };
}
