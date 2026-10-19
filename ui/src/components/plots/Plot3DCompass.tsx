import { compassOctant, type CameraOrientation } from "@/lib/plot/cameraOrientation";
import { usePaneContext } from "./paneContext";
import "./Plot3DCompass.css";

/* SVG compass rose inset: azimuth needle + elevation readout + optional storm-bearing marker. */

// SVG viewport: a square of this size in logical pixels.
const VIEW_SIZE = 64;
// Centre of the SVG coordinate space.
const CX = VIEW_SIZE / 2;
const CY = VIEW_SIZE / 2;
// Outer radius of the compass ring.
const RING_R = 28;
// Length of a minor tick from the ring inward.
const TICK_MINOR = 4;
// Length of a cardinal tick from the ring inward.
const TICK_CARDINAL = 7;
// Needle dimensions (half-width, head length, tail length).
const NEEDLE_HALF_W = 3;
const NEEDLE_HEAD_LEN = 16;
const NEEDLE_TAIL_LEN = 10;
// Storm marker: half the diagonal of the open diamond.
const STORM_DIAMOND_R = 5;

/** Compass bearing (0=N, CW) to SVG rotate() degrees (identity: needle starts pointing up). */
function bearingToSvgDeg(bearing: number): number {
  return bearing;
}

/** Render a needle polygon pointing upward from centre; rotate by bearing. */
function NeedleHead({ bearing }: { bearing: number }) {
  // Triangle tip at (CX, CY - RING_R * 0.85), base at (CX ± w, CY + tail / 2).
  const tipY = CY - NEEDLE_HEAD_LEN;
  const baseY = CY;
  const points = [
    `${CX},${tipY}`,
    `${CX - NEEDLE_HALF_W},${baseY}`,
    `${CX + NEEDLE_HALF_W},${baseY}`,
  ].join(" ");
  return (
    <polygon
      className="plot3d-compass__needle-head"
      points={points}
      transform={`rotate(${bearingToSvgDeg(bearing)} ${CX} ${CY})`}
    />
  );
}

/** Tail half of the needle (pointing away from the bearing direction). */
function NeedleTail({ bearing }: { bearing: number }) {
  const tipY = CY + NEEDLE_TAIL_LEN;
  const baseY = CY;
  const points = [
    `${CX},${tipY}`,
    `${CX - NEEDLE_HALF_W},${baseY}`,
    `${CX + NEEDLE_HALF_W},${baseY}`,
  ].join(" ");
  return (
    <polygon
      className="plot3d-compass__needle-tail"
      points={points}
      transform={`rotate(${bearingToSvgDeg(bearing)} ${CX} ${CY})`}
    />
  );
}

/** Small open diamond at a given compass bearing from the centre. */
function StormMarker({ bearing }: { bearing: number }) {
  // Diamond centred on the ring at the bearing direction.
  const r = RING_R - STORM_DIAMOND_R - 2;
  const points = [
    `0,${-STORM_DIAMOND_R}`,
    `${STORM_DIAMOND_R},0`,
    `0,${STORM_DIAMOND_R}`,
    `${-STORM_DIAMOND_R},0`,
  ].join(" ");
  return (
    <polygon
      className="plot3d-compass__storm-marker"
      points={points}
      transform={`rotate(${bearingToSvgDeg(bearing)} ${CX} ${CY}) translate(${CX} ${CY - r})`}
    />
  );
}

/**
 * Compass ticks: 16 minor ticks + 4 cardinal ticks, arranged around the ring.
 */
function CompassTicks() {
  const ticks: React.ReactNode[] = [];
  for (let i = 0; i < 16; i++) {
    const angleDeg = i * 22.5;
    const isCardinal = i % 4 === 0;
    const tickLen = isCardinal ? TICK_CARDINAL : TICK_MINOR;
    const rad = (angleDeg - 90) * (Math.PI / 180);
    const outerX = CX + RING_R * Math.cos(rad);
    const outerY = CY + RING_R * Math.sin(rad);
    const innerX = CX + (RING_R - tickLen) * Math.cos(rad);
    const innerY = CY + (RING_R - tickLen) * Math.sin(rad);
    ticks.push(
      <line
        key={i}
        className={`plot3d-compass__tick${isCardinal ? " plot3d-compass__tick--cardinal" : ""}`}
        x1={outerX}
        y1={outerY}
        x2={innerX}
        y2={innerY}
      />,
    );
  }
  return <>{ticks}</>;
}

function fmtElevation(deg: number): string {
  const sign = deg >= 0 ? "+" : "";
  return `${sign}${deg.toFixed(0)}°`;
}

function fmtRange(metres: number): string {
  if (metres >= 1000) return `${(metres / 1000).toFixed(1)} km`;
  return `${metres.toFixed(0)} m`;
}

export interface Plot3DCompassProps {
  /** Current camera orientation in true metres, from useCameraOrientation. */
  orientation: CameraOrientation | null;
}

/**
 * Inset compass rose for a 3D scene. Returns null before camera is ready or when
 * this pane does not own shared overlays (locked grid: only one compass per bearing group).
 */
export function Plot3DCompass({ orientation }: Plot3DCompassProps) {
  const { ownsSharedOverlays } = usePaneContext();
  if (!orientation) return null;
  if (!ownsSharedOverlays) return null;

  const octant = compassOctant(orientation.azimuthDeg);
  const hasStorm =
    orientation.stormBearingDeg !== undefined &&
    orientation.stormRangeM !== undefined;

  return (
    <div className="plot3d-compass" aria-hidden="true">
      <svg
        className="plot3d-compass__rose"
        viewBox={`0 0 ${VIEW_SIZE} ${VIEW_SIZE}`}
        aria-hidden="true"
      >
        {/* Background ring */}
        <circle
          className="plot3d-compass__ring"
          cx={CX}
          cy={CY}
          r={RING_R}
        />

        {/* Tick marks */}
        <CompassTicks />

        {/* N label — always at the top */}
        <text
          className="plot3d-compass__north-label"
          x={CX}
          y={CY - RING_R + TICK_CARDINAL + 5}
        >
          N
        </text>

        {/* Storm bearing marker (behind needle so needle reads over it) */}
        {hasStorm && (
          <StormMarker bearing={orientation.stormBearingDeg!} />
        )}

        {/* Viewing-direction needle */}
        <NeedleTail bearing={orientation.azimuthDeg} />
        <NeedleHead bearing={orientation.azimuthDeg} />
      </svg>

      <div className="plot3d-compass__readout">
        <span className="plot3d-compass__azimuth">
          {orientation.azimuthDeg.toFixed(0)}° {octant}
        </span>
        <span className="plot3d-compass__elevation">
          el {fmtElevation(orientation.elevationDeg)}
        </span>
        {hasStorm && (
          <span className="plot3d-compass__storm-range">
            storm {fmtRange(orientation.stormRangeM!)}
          </span>
        )}
      </div>
    </div>
  );
}
