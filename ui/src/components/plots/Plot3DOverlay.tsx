import { useState, type ReactNode } from "react";
import { Icon } from "@/components/shell/Icon";
import type { StepInfo } from "@/lib/run/runOutputs";
import { cssColorbarGradient } from "@/lib/plot/plotColormaps";
import {
  describeVtkSampling,
  renderFidelityLabel,
  type RenderFidelityMode,
  type VtkSamplingProvenance,
} from "@/lib/plot/renderFidelity";
import { describeDomainGrid } from "@/lib/plot/domainAxes";
import type { ProjectedAxisLabel } from "./useDomainAxes";
import "./Plot3DOverlay.css";

/* HTML overlays for VTK plots. Data labels stay pointer-transparent; the scale
 * badge owns one small disclosure button so its footprint can be reclaimed. */

/** Stacking container for top-right overlays (colorbar, legend, compass) — prevents corner overlap. */
export function Plot3DOverlayStack({ children }: { children: ReactNode }) {
  return <div className="plot3d-overlay-stack">{children}</div>;
}

const BADGE_COLLAPSED_KEY = "nimbus-studio:plot3d-provenance-collapsed:v1";

function initialBadgeCollapsed(): boolean {
  if (typeof window === "undefined") return false;
  return window.localStorage.getItem(BADGE_COLLAPSED_KEY) === "true";
}

/** Physical domain extent or source sample counts, plus display provenance. */
export function Plot3DScaleBadge({
  grid,
  zExag,
  detail,
  fidelityMode,
  samplingProvenance,
  missingSupport,
}: {
  grid: StepInfo["grid"] | null | undefined;
  /** Vertical exaggeration to report; omit where the view doesn't track it. */
  zExag?: number;
  /** Optional visualization provenance, such as coarse-grid reconstruction. */
  detail?: string | null;
  /** Shared resolved-only vs disclosed-presentation contract. */
  fidelityMode?: RenderFidelityMode;
  /** Worker-recorded native -> sampled -> displayed lattice history. */
  samplingProvenance?: VtkSamplingProvenance | null;
  missingSupport?: boolean;
}) {
  const [collapsed, setCollapsed] = useState(initialBadgeCollapsed);
  const domain = describeDomainGrid(grid);
  if (!domain) return null;
  const sub = [
    domain.notice,
    zExag != null ? `vert ×${zExag.toFixed(1)}` : null,
    ...describeVtkSampling(samplingProvenance),
    detail,
    missingSupport ? "missing regions and unsupported interpolation omitted" : null,
  ].filter(Boolean);
  const toggleCollapsed = () => {
    setCollapsed((current) => {
      const next = !current;
      try {
        window.localStorage.setItem(BADGE_COLLAPSED_KEY, String(next));
      } catch {
        // The current figure still responds when persistence is unavailable.
      }
      return next;
    });
  };
  if (collapsed) {
    return (
      <div className="plot3d-badge plot3d-badge--collapsed">
        <button
          type="button"
          className="plot3d-badge__toggle"
          aria-label="Show scientific resolution details"
          title="Show scientific resolution details"
          onClick={toggleCollapsed}
        >
          <Icon name="info" size={13} />
        </button>
      </div>
    );
  }
  return (
    <div className="plot3d-badge">
      <button
        type="button"
        className="plot3d-badge__toggle"
        aria-label="Hide scientific resolution details"
        title="Hide scientific resolution details"
        onClick={toggleCollapsed}
      >
        <Icon name="chevron-down" size={12} />
      </button>
      {fidelityMode && (
        <span className={`plot3d-badge__mode plot3d-badge__mode--${fidelityMode}`}>
          {renderFidelityLabel(fidelityMode)}
        </span>
      )}
      <span>
        {domain.caption}
      </span>
      {sub.length > 0 && <span className="plot3d-badge__sub">{sub.join(" · ")}</span>}
    </div>
  );
}

function fmtTick(v: number): string {
  const a = Math.abs(v);
  if (a !== 0 && (a < 0.01 || a >= 1e5)) return v.toExponential(1);
  return Number(v.toPrecision(3)).toString();
}

/** Vertical colour scale (gradient + min/max + label), top-right. */
export function Plot3DColorbar({
  colormap,
  min,
  max,
  label,
  unit,
}: {
  colormap: string;
  min: number;
  max: number;
  label?: string | null;
  unit?: string | null;
}) {
  if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
  const title = label ? `${label}${unit ? ` (${unit})` : ""}` : (unit ?? "");
  return (
    <div className="plot3d-colorbar">
      {title && <div className="plot3d-colorbar__title">{title}</div>}
      <div className="plot3d-colorbar__body">
        <div
          className="plot3d-colorbar__bar"
          data-vtk-colormap={colormap}
          style={{ background: cssColorbarGradient(colormap) }}
        />
        <div className="plot3d-colorbar__ticks">
          <span>{fmtTick(max)}</span>
          <span>{fmtTick(min)}</span>
        </div>
      </div>
    </div>
  );
}


/** Discrete category/value legend for paired surfaces, top-right. */
export function Plot3DDiscreteLegend({
  title,
  items,
}: {
  title?: string | null;
  items: ReadonlyArray<{ color: string; label: string }>;
}) {
  if (items.length === 0) return null;
  return (
    <div className="plot3d-discrete-legend">
      {title && <div className="plot3d-discrete-legend__title">{title}</div>}
      {items.map((item) => (
        <div className="plot3d-discrete-legend__item" key={`${item.color}:${item.label}`}>
          <span
            className="plot3d-discrete-legend__swatch"
            style={{ background: item.color }}
          />
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  );
}

/** DOM axis tick labels projected from useDomainAxes (vtk.js text renders nothing; positions are CSS %). */
export function Plot3DAxisLabels({ labels }: { labels: ProjectedAxisLabel[] }) {
  if (labels.length === 0) return null;
  return (
    <div className="plot3d-axis-labels">
      {labels.map((label) => (
        <span
          key={label.key}
          className={`plot3d-axis-labels__item plot3d-axis-labels__item--${label.kind}`}
          style={{ left: `${label.leftPct}%`, top: `${label.topPct}%` }}
        >
          {label.text}
        </span>
      ))}
    </div>
  );
}
