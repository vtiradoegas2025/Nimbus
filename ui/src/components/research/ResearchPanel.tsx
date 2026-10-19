import { memo, useMemo } from "react";
import type { Layout } from "plotly.js";
import { PlotCanvas } from "@/components/studio/PlotCanvas";
import type { ResearchPanel as Panel } from "./common";
export const ResearchPanelView = memo(function ResearchPanelView({ panel, index, range, identity, onViewChange, generation, onRendered }: {
    panel: Panel;
    index: number;
    range?: Partial<Layout>;
    identity: string;
    onViewChange: (panel: Panel, event: Readonly<Record<string, unknown>>) => void;
    generation: number;
    onRendered: (id: string, generation: number) => void;
}) {
    const layout = useMemo<Partial<Layout>>(() => ({ ...panel.layout, ...range, margin: { l: 65, r: 55, b: 65, ...panel.layout?.margin, t: 95 }, legend: { orientation: "h", x: 0, y: 1.15, yanchor: "bottom", font: { size: 10 } }, xaxis: { ...panel.layout?.xaxis, ...range?.xaxis }, yaxis: { ...panel.layout?.yaxis, ...range?.yaxis },
        ...(panel.layout?.scene ? { scene: { ...panel.layout.scene, ...range?.scene } } : {}) }), [panel.layout, range]);
    const data = useMemo(() => panel.data.map(trace => "colorbar" in trace && trace.colorbar ? { ...trace, colorbar: { ...trace.colorbar, thickness: 14, title: { ...(typeof trace.colorbar.title === "object" ? trace.colorbar.title : { text: trace.colorbar.title }), side: "right" as const } } } : trace), [panel.data]);
    return <figure className="research-panel" data-panel-id={panel.id}><figcaption><strong>{String.fromCharCode(65 + index)}.</strong> {panel.title}</figcaption><PlotCanvas data={data} config={panel.config} layout={layout} className="research-canvas" viewIdentity={identity} onClick={panel.onClick} onRendered={() => onRendered(panel.id, generation)} onRelayout={event => onViewChange(panel, event)}/></figure>;
});
