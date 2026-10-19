import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import Plotly from "plotly.js-dist-min";
import type { Data, Layout, Config, LayoutAxis } from "plotly.js";
import {
  currentStudioAppearance,
  STUDIO_APPEARANCE_APPLIED_EVENT,
} from "@/lib/studio/studioAppearance";
import { acceleratePlotlyData } from "@/lib/plot/plotAcceleration";
import { smoothHeatmapDisplay } from "@/lib/plot/heatmapDisplay";
import { plotViewerPixelRatio } from "@/lib/plot/plotPixelRatio";
import { beginPlotCanvasOperation, retirePlotCanvas } from "@/lib/plot/plotCanvasReadiness";
import "./PlotCanvas.css";

/* Thin Plotly.js wrapper. Uses Plotly.react (no recreation on update) and a ResizeObserver for flex containers. */
interface PlotCanvasProps {
  data: Data[];
  layout?: Partial<Layout>;
  config?: Partial<Config>;
  className?: string;
  viewIdentity?: string;
  /** Fires after the latest render settles; generation-gated so stale renders cannot advance frame identity. */
  onRendered?: () => void;
  /** Plotly viewport changes (zoom/pan/autoscale); binds calculations to the visible coordinate box. */
  onRelayout?: (event: Readonly<Record<string, unknown>>) => void;
  /** Point selection; callers interpret the Plotly payload in their coordinate system. */
  onClick?: (event: Readonly<Record<string, unknown>>) => void;
}

interface PlotlyEventTarget extends HTMLDivElement {
  on?: (event: "plotly_relayout" | "plotly_click", handler: (payload: Record<string, unknown>) => void) => void;
  removeListener?: (event: "plotly_relayout" | "plotly_click", handler: (payload: Record<string, unknown>) => void) => void;
}

function accessiblePlotLabel(layout: Partial<Layout> | undefined): string {
  const title = layout?.title;
  const rawTitle = typeof title === "string"
    ? title
    : title && typeof title === "object" && "text" in title
      ? title.text
      : undefined;
  if (typeof rawTitle !== "string") return "Interactive scientific plot";
  const plainTitle = rawTitle.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  return plainTitle ? `Interactive scientific plot: ${plainTitle}` : "Interactive scientific plot";
}

const DEFAULT_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: {
    family: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    size: 11,
  },
  margin: { l: 56, r: 24, t: 64, b: 48 },
  // Keep zoom across frame updates; a changed view identity resets the coordinate system.
  uirevision: "nimbus",
};

const DEFAULT_CONFIG: Partial<Config> = {
  responsive: true,
  displaylogo: false,
  /*
   * Export at 3x, not at screen resolution.
   * The camera button produced a PNG the size of the on-screen figure — roughly 100 DPI once
   * placed in a document, which is below what a journal or a projected talk needs. Plotly's `scale`
   * re-renders the vector scene at the higher resolution rather than upsampling pixels, so text and
   * axes stay sharp; 3x on a ~1100 px figure lands near 300 DPI at single-column width.
   */
  toImageButtonOptions: { format: "png", filename: "nimbus-plot", scale: 3 },
  displayModeBar: true, // Always visible; lasso/select removed as irrelevant for scientific heatmaps.
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
  /*
   * A second download, in vector.
   *
   * A raster PNG is fixed at whatever resolution it was written at; a figure headed for a paper
   * wants type and axes that stay sharp at any size the typesetter picks. Heatmap cells still
   * rasterise inside the SVG — Plotly draws them as an embedded image — but the axes, ticks,
   * colorbar and every label remain text and paths, which is what a reviewer zooms into.
   */
  modeBarButtonsToAdd: [{
    name: "downloadSvg",
    title: "Download as SVG (vector, for publication)",
    icon: Plotly.Icons.disk,
    click: (graphDiv: Plotly.PlotlyHTMLElement) => {
      // Sized from the figure on screen: Plotly requires explicit dimensions here, and a
      // hard-coded pair would letterbox or crop whatever the current aspect happens to be.
      const box = graphDiv.getBoundingClientRect();
      void Plotly.downloadImage(graphDiv, {
        format: "svg",
        filename: "nimbus-plot",
        width: Math.max(1, Math.round(box.width)),
        height: Math.max(1, Math.round(box.height)),
      });
    },
  }],
  scrollZoom: true, // Scroll-to-zoom; double-click or autoscale button resets the view.
};

export function PlotCanvas({ data, layout, config, className, viewIdentity, onRendered, onRelayout, onClick }: PlotCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const initializedRef = useRef(false);
  const renderGenerationRef = useRef(0);
  const operationQueueRef = useRef<Promise<void>>(Promise.resolve());
  const disposedRef = useRef(false);
  const onRenderedRef = useRef(onRendered);
  const onRelayoutRef = useRef(onRelayout);
  const onClickRef = useRef(onClick);
  const viewIdentityRef = useRef(viewIdentity);
  const renderedViewIdentityRef = useRef(viewIdentity);
  viewIdentityRef.current = viewIdentity;
  const relayoutAttachedRef = useRef(false);
  const [appearanceRevision, setAppearanceRevision] = useState(0);
  const [smoothHeatmaps, setSmoothHeatmaps] = useState(false);
  const [viewerPixelRatio, setViewerPixelRatio] = useState(() => plotViewerPixelRatio(window.devicePixelRatio));
  const hasHeatmaps = useMemo(() => data.some(trace => trace.type === "heatmap"), [data]);
  const heatmapDisplay = useMemo(() => smoothHeatmapDisplay(data, smoothHeatmaps, layout), [data, smoothHeatmaps, layout]);
  const relayoutHandlerRef = useRef((payload: Record<string, unknown>) => {
    if (renderedViewIdentityRef.current !== viewIdentityRef.current) return;
    onRelayoutRef.current?.(payload);
  });
  const clickHandlerRef = useRef((payload: Record<string, unknown>) => {
    onClickRef.current?.(payload);
  });
  onRelayoutRef.current = onRelayout;
  onClickRef.current = onClick;

  useLayoutEffect(() => {
    onRenderedRef.current = onRendered;
  }, [onRendered]);

  useEffect(() => {
    const onAppearance = () => setAppearanceRevision((revision) => revision + 1);
    window.addEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
    return () => window.removeEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
  }, []);

  useEffect(() => {
    const update = () => setViewerPixelRatio(plotViewerPixelRatio(window.devicePixelRatio));
    const density = window.matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`);
    window.addEventListener("resize", update);
    density.addEventListener("change", update);
    update();
    return () => {
      window.removeEventListener("resize", update);
      density.removeEventListener("change", update);
    };
  }, [viewerPixelRatio]);

  // Plotly owns an SVG subtree and cannot consume CSS variables directly. Its
  // bundled colour parser also rejects CSS Color 4 oklch() strings and silently
  // falls back to #444, which is nearly invisible on the dark plot paper.
  // --plot-text is therefore deliberately stored as rgb/hex in theme.css.
  const chrome = useMemo(() => {
    void appearanceRevision;
    const light = currentStudioAppearance() === "light";
    const style = typeof document === "undefined" ? null : getComputedStyle(document.documentElement);
    const token = (name: string, fallback: string) => style?.getPropertyValue(name).trim() || fallback;
    return {
      text: token("--plot-text", light ? "#4d5864" : "#a3a8b3"),
      grid: token("--plot-gridline", light ? "rgba(24,26,30,0.1)" : "rgba(255,255,255,0.04)"),
      zero: token("--plot-zeroline", light ? "rgba(24,26,30,0.22)" : "#2e3340"),
      overlay: token("--plot-overlay-bg", light ? "rgba(253,251,247,0.82)" : "rgba(20,23,29,0.7)"),
      textScale: typeof document !== "undefined" && document.documentElement.dataset.textSize === "comfortable" ? 13 / 11 : 1,
      ink: token("--plot-ink", light ? "rgba(24,26,30,0.92)" : "#eef1f7"),
    };
  }, [appearanceRevision]);
  const mergedLayout = useMemo<Partial<Layout>>(() => {
    const scaledFont = (font: { size?: number } | undefined, fallback: number) => ({ ...font, size: (font?.size ?? fallback) * chrome.textScale });
    const axes: Record<string, Partial<LayoutAxis>> = {};
    const axisNames = new Set(["xaxis", "yaxis", ...Object.keys(layout ?? {}).filter((key) => /^[xy]axis\d*$/.test(key))]);
    for (const name of axisNames) {
      const axis = (layout as Record<string, Partial<LayoutAxis>> | undefined)?.[name];
      axes[name] = {
        ...(axis?.color == null ? { gridcolor: chrome.grid, zerolinecolor: chrome.zero } : {}),
        automargin: true,
        ...axis,
        ...(axis?.tickfont ? { tickfont: scaledFont(axis.tickfont, 11) } : {}),
        ...(axis?.title && typeof axis.title !== "string" ? { title: { ...axis.title, font: scaledFont(axis.title.font, 13) } } : {}),
      };
    }
    const title = typeof layout?.title === "string" ? { text: layout.title } : layout?.title;
    return {
      ...DEFAULT_LAYOUT,
      ...layout,
      ...(viewIdentity === undefined ? {} : { uirevision: viewIdentity }),
      ...axes,
      title: title ? { yref: "paper", y: 1, yanchor: "bottom", pad: { b: 8 }, ...title, font: scaledFont(title.font, 13) } : undefined,
      modebar: { bgcolor: chrome.overlay, color: chrome.text, activecolor: chrome.ink, ...layout?.modebar },
      font: { ...DEFAULT_LAYOUT.font, color: chrome.text, ...layout?.font, size: (layout?.font?.size ?? 11) * chrome.textScale },
      annotations: layout?.annotations?.map((annotation) => ({ ...annotation, font: scaledFont(annotation.font, 12) })),
    };
  }, [layout, chrome, viewIdentity]);
  const mergedConfig = useMemo<Partial<Config>>(
    () => ({ ...DEFAULT_CONFIG, plotGlPixelRatio: viewerPixelRatio, ...config,
      modeBarButtonsToAdd: [
        ...(config?.modeBarButtonsToAdd ?? DEFAULT_CONFIG.modeBarButtonsToAdd ?? []),
        ...(hasHeatmaps ? [{
          name: "heatmapDisplay",
          title: smoothHeatmaps
            ? heatmapDisplay.smoothed > 0 ? "Restore original shading (smooth display active)" : "Original shading retained: gaps or nonuniform axes"
            : "Smooth heatmap shading (display only; values unchanged)",
          icon: { width: 20, height: 20, path: "M2 2h7v7H2zM11 2h7v7h-7zM2 11h7v7H2zM11 11h7v7h-7z" },
          click: () => setSmoothHeatmaps(value => !value),
        }] : []),
      ],
    }),
    [config, viewerPixelRatio, hasHeatmaps, smoothHeatmaps, heatmapDisplay.smoothed],
  );
  const acceleratedPlot = useMemo(() => acceleratePlotlyData(heatmapDisplay.data), [heatmapDisplay.data]);
  const plotLabel = useMemo(() => accessiblePlotLabel(layout), [layout]);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const generation = ++renderGenerationRef.current;
    const finishRender = beginPlotCanvasOperation(el);
    // Completion belongs to these traces, not to a newer frame staged while Plotly is drawing.
    const reportRendered = onRenderedRef.current;
    const scheduled = operationQueueRef.current
      .catch(() => undefined)
      .then(async () => {
        // Coalesce to latest props; never call into a graph div that unmount has retired.
        if (disposedRef.current || renderGenerationRef.current !== generation) return;
        if (!initializedRef.current) {
          await Plotly.newPlot(el, acceleratedPlot.data, mergedLayout, mergedConfig);
          initializedRef.current = true;
        } else {
          await Plotly.react(el, acceleratedPlot.data, mergedLayout, mergedConfig);
        }
        const plotlyTarget = el as PlotlyEventTarget;
        if (!relayoutAttachedRef.current && plotlyTarget.on) {
          plotlyTarget.on("plotly_relayout", relayoutHandlerRef.current);
          plotlyTarget.on("plotly_click", clickHandlerRef.current);
          relayoutAttachedRef.current = true;
        }
        if (!disposedRef.current && renderGenerationRef.current === generation) {
          renderedViewIdentityRef.current = viewIdentity;
          reportRendered?.();
        }
      });
    operationQueueRef.current = scheduled.then(() => finishRender()).catch((error: unknown) => {
      finishRender(error instanceof Error ? error : new Error(String(error)));
      // Stale/unmounted work is deliberately discarded. A current Plotly
      // failure still surfaces as a page error instead of being hidden.
      if (!disposedRef.current && renderGenerationRef.current === generation) {
        queueMicrotask(() => { throw error; });
      }
    });
    // Workspace chrome (pinning a dataset, opening a tool, status text) can
    // re-render PlotStudio without changing a plot. Plotly.react is not a
    // harmless no-op in that case: it rebuilds renderer state and WebKit can
    // exhaust/crash its content process under repeated unrelated updates.
    // Call it only for a real plot/config/appearance change. ResizeObserver
    // below independently handles changes to the canvas box.
  }, [acceleratedPlot.data, mergedConfig, mergedLayout, viewIdentity]);

  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const relayoutHandler = relayoutHandlerRef.current;
    const clickHandler = clickHandlerRef.current;
    disposedRef.current = false;
    /*
     * The observed element is the one Plotly.Plots.resize redraws, so an unfiltered handler can feed
     * itself. Under a locked aspect that is not hypothetical: `constrain: "domain"` shrinks the plot
     * area to hold 1:1, which is a no-op on a square plan view but a large change on a tall
     * cross-section, and the redraw can report a new box that schedules another resize. Acting only
     * on a box we have not already resized at breaks the cycle without gating legitimate resizes.
     */
    let lastWidth = -1;
    let lastHeight = -1;
    let resizeQueued = false;
    const observer = new ResizeObserver((entries) => {
      const box = entries[entries.length - 1]?.contentRect;
      if (box) {
        const width = Math.round(box.width);
        const height = Math.round(box.height);
        if (width === lastWidth && height === lastHeight) return;
        lastWidth = width;
        lastHeight = height;
      }
      // Resize reads live dimensions; keep only one waiting behind Plotly's in-flight work.
      if (resizeQueued) return;
      resizeQueued = true;
      const finishResize = beginPlotCanvasOperation(el, "resize");
      operationQueueRef.current = operationQueueRef.current
        .catch(() => undefined)
        .then(async () => {
          resizeQueued = false;
          // Not generation-gated: resize is idempotent and always correct; gating caused stale sizes during playback.
          if (
            disposedRef.current
            || !initializedRef.current
            || !el.isConnected
            || el.getClientRects().length === 0
            || el.clientWidth === 0
            || el.clientHeight === 0
          ) return;
          try {
            await Plotly.Plots.resize(el);
          } catch (error) {
            /*
             * A pane can become hidden between the geometry check and Plotly's
             * own check (notably while switching run sources on Retina WebKit).
             * Skipping that obsolete resize is safe: ResizeObserver schedules
             * a fresh one when the pane is displayed again.
             */
            if (
              error instanceof Error
              && error.message === "Resize must be passed a displayed plot div element."
            ) return;
            throw error;
          }
        }).then(() => finishResize(), (error: unknown) => {
          finishResize(error instanceof Error ? error : new Error(String(error)));
          throw error;
        });
    });
    observer.observe(el);
    return () => {
      disposedRef.current = true;
      retirePlotCanvas(el);
      observer.disconnect();
      renderGenerationRef.current += 1;
      // Purge after the queue drains; purging mid-promise produces `_t.emit is not a function`.
      operationQueueRef.current = operationQueueRef.current
        .catch(() => undefined)
        .then(() => {
          const plotlyTarget = el as PlotlyEventTarget;
          if (relayoutAttachedRef.current) {
            plotlyTarget.removeListener?.("plotly_relayout", relayoutHandler);
            plotlyTarget.removeListener?.("plotly_click", clickHandler);
            relayoutAttachedRef.current = false;
          }
          Plotly.purge(el);
          initializedRef.current = false;
        });
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className={`plot-canvas ${className ?? ""}`}
      role="region"
      aria-label={plotLabel}
      data-render-backend={acceleratedPlot.backend}
      data-accelerated-traces={acceleratedPlot.acceleratedTraceCount}
      data-plot-points={acceleratedPlot.pointCount}
      data-heatmap-display={heatmapDisplay.smoothed > 0 ? "smooth" : "native"}
      data-display-pixel-ratio={mergedConfig.plotGlPixelRatio}
    />
  );
}
