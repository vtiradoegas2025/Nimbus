/*
 * `plotly.js-dist-min` is the pre-bundled minified Plotly distribution. It
 * has no types of its own, but the surface matches `plotly.js`. We alias
 * the module declaration here so the import type-checks against the
 * @types/plotly.js definitions.
 */
declare module "plotly.js-dist-min" {
  import type Plotly from "plotly.js";
  const PlotlyExport: typeof Plotly;
  export default PlotlyExport;
}
