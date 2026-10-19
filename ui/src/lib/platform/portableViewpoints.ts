export const PORTABLE_VIEWPOINT_MAX_BYTES = 64 * 1024 * 1024;
export const PORTABLE_VIEWPOINT_COUNTS = [12, 24, 36, 72] as const;
export interface PortableViewpoint { angleDeg: number; png: string }
export interface PortableViewpointDocument {
  schema: "nimbus.portable-viewpoints";
  version: 1;
  title: string;
  width: number;
  height: number;
  source: unknown;
  scene: unknown;
  views: PortableViewpoint[];
}
export function orbitPoint(point: readonly number[], center: readonly number[], degrees: number): [number, number, number] {
  if (point.length !== 3 || center.length !== 3 || ![...point, ...center, degrees].every(Number.isFinite)) throw new Error("Invalid orbit camera");
  const angle = degrees * Math.PI / 180, c = Math.cos(angle), s = Math.sin(angle);
  const x = point[0] - center[0], y = point[1] - center[1];
  return [center[0] + x * c - y * s, center[1] + x * s + y * c, point[2]];
}
export function buildPortableViewpoints(document: PortableViewpointDocument): Uint8Array {
  if (!document.views.length || document.views.length > 72 || document.views.some(view => !Number.isFinite(view.angleDeg)
    || !/^data:image\/png;base64,[A-Za-z0-9+/]+=*$/.test(view.png))) throw new Error("Invalid portable viewpoints");
  const payload = JSON.stringify(document).replace(/</g, "\\u003c");
  if (payload.length > PORTABLE_VIEWPOINT_MAX_BYTES) throw new Error("Portable viewer exceeds 64 MiB; choose fewer viewpoints or a smaller viewport");
  const html = `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'"><title>Nimbus portable viewpoints</title>
<style>body{margin:0;background:#0e1117;color:#e6e9ef;font:15px system-ui,sans-serif}header,details{padding:16px 24px}h1{font-size:20px;margin:0 0 8px}p{margin:6px 0;color:#b9c2d0}nav{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 24px;border-block:1px solid #394150}button,input{font:inherit}button{padding:8px 12px;cursor:pointer}label{display:flex;align-items:center;gap:8px}#viewport{overflow:auto;height:65vh;min-height:240px;touch-action:pan-y}img{display:block;margin:auto;max-width:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer}button:focus-visible,input:focus-visible,summary:focus-visible{outline:3px solid #89bcff;outline-offset:3px}</style>
<header><h1 id="title"></h1><p>Pre-rendered viewpoints at one saved time. These are images, not a free-rotation 3D model or numerical dataset.</p><p id="description"></p></header>
<nav aria-label="View controls"><button id="prev" aria-label="Previous viewpoint">←</button><label>Viewpoint <input id="angle" type="range" min="0" value="0"></label><button id="next" aria-label="Next viewpoint">→</button><output id="position" aria-live="polite"></output><label>Magnification <input id="zoom" type="range" min="25" max="200" step="25" value="100"></label><output id="zoom-label">100%</output><button id="fit">Fit image</button></nav>
<div id="viewport" tabindex="0" aria-label="Captured plot; use left and right arrows to change viewpoint"><img id="image" alt="Captured scientific plot"></div><details><summary>Source and scene metadata</summary><pre id="metadata"></pre></details>
<script id="nimbus-viewpoints" type="application/json">${payload}</script>
<script>(()=>{const data=JSON.parse(document.getElementById('nimbus-viewpoints').textContent);const image=document.getElementById('image'),angle=document.getElementById('angle'),zoom=document.getElementById('zoom'),viewport=document.getElementById('viewport');angle.max=data.views.length-1;document.getElementById('title').textContent=data.title;document.getElementById('description').textContent=data.views.length+' lossless PNG views · '+data.width+' × '+data.height+' captured pixels · fixed elevation. Magnification adds no source detail. Works offline; only the selected image is displayed.';const {views,...metadata}=data;document.getElementById('metadata').textContent=JSON.stringify(metadata,null,2);function show(){const i=Number(angle.value),view=data.views[i];image.src=view.png;image.alt='Scientific plot, viewpoint '+(i+1)+' of '+data.views.length+', '+view.angleDeg+' degrees from opening camera';document.getElementById('position').textContent=(i+1)+' / '+data.views.length+' · '+view.angleDeg+'°';}function scale(){image.style.width=(data.width*Number(zoom.value)/100)+'px';document.getElementById('zoom-label').textContent=zoom.value+'%';}function step(delta){angle.value=(Number(angle.value)+delta+data.views.length)%data.views.length;show();}angle.addEventListener('input',show);zoom.addEventListener('input',scale);document.getElementById('prev').onclick=()=>step(-1);document.getElementById('next').onclick=()=>step(1);document.getElementById('fit').onclick=()=>{image.style.width=Math.min(data.width,viewport.clientWidth)+'px';document.getElementById('zoom-label').textContent=Math.round(Math.min(data.width,viewport.clientWidth)/data.width*100)+'%';};viewport.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}};show();scale();})();</script></html>`;
  const bytes = new TextEncoder().encode(html);
  if (bytes.length > PORTABLE_VIEWPOINT_MAX_BYTES) throw new Error("Portable viewer exceeds 64 MiB; choose fewer viewpoints or a smaller viewport");
  return bytes;
}
