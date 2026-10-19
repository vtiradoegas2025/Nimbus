import { useMemo } from "react";
import { PlotCanvas } from "@nimbus/components/studio/PlotCanvas";
import type { Data, Layout } from "plotly.js";

export function PlotResolutionHarness() {
  const data = useMemo<Data[]>(() => [{ type: "scatter", mode: "lines",
    x: Array.from({ length: 6000 }, (_, index) => index / 100),
    y: Array.from({ length: 6000 }, (_, index) => Math.sin(index / 100)), line: { width: 2 } }], []);
  const layout = useMemo<Partial<Layout>>(() => ({ title: { text: "Manufactured dense line" },
    xaxis: { title: { text: "x" } }, yaxis: { title: { text: "saved samples" } }, margin: { l: 75, r: 25, t: 80, b: 55 } }), []);
  return <main style={{ width: "100vw", height: "100vh", display: "flex", flexDirection: "column" }}>
    <div className="plot-body"><PlotCanvas data={data} layout={layout} /></div>
  </main>;
}
