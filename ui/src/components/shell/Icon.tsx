/* Minimal hand-rolled SVG icon set. 16x16 viewBox, stroke="currentColor". */
export type IconName =
  | "play"
  | "pause"
  | "record"
  | "queue"
  | "archive"
  | "package"
  | "plot"
  | "settings"
  | "chevron-right"
  | "chevron-down"
  | "chevron-up"
  | "info"
  | "warning"
  | "dot"
  | "plus"
  | "stop"
  | "image"
  | "x"
  | "folder"
  | "lock"
  | "undo"
  | "redo"
  | "refresh";

interface IconProps {
  name: IconName;
  size?: number;
  className?: string;
}

export function Icon({ name, size = 16, className }: IconProps) {
  const path = ICONS[name];
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {path}
    </svg>
  );
}

const ICONS: Record<IconName, JSX.Element> = {
  play: <path d="M4 3l9 5-9 5V3z" fill="currentColor" stroke="none" />,
  pause: (
    <>
      <rect x="4" y="3.5" width="3" height="9" rx="0.5" fill="currentColor" stroke="none" />
      <rect x="9" y="3.5" width="3" height="9" rx="0.5" fill="currentColor" stroke="none" />
    </>
  ),
  record: <circle cx="8" cy="8" r="4" fill="currentColor" stroke="none" />,
  stop: <rect x="4" y="4" width="8" height="8" rx="1" fill="currentColor" stroke="none" />,
  queue: (
    <>
      <path d="M2.5 4h11" />
      <path d="M2.5 8h11" />
      <path d="M2.5 12h7" />
    </>
  ),
  archive: (
    <>
      <rect x="2" y="3.5" width="12" height="3" rx="0.5" />
      <path d="M3 6.5v6a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-6" />
      <path d="M6.5 9h3" />
    </>
  ),
  package: (
    <>
      <path d="M8 1.75 14 5v6L8 14.25 2 11V5z" />
      <path d="M2 5l6 3.25L14 5" />
      <path d="M8 8.25v6" />
    </>
  ),
  plot: (
    <>
      <path d="M2.5 13.5V3" />
      <path d="M2.5 13.5h11" />
      <path d="M5 11l2-3 2 2 3-5" />
    </>
  ),
  settings: (
    <>
      <circle cx="8" cy="8" r="2" />
      <path d="M8 1v2M8 13v2M14.06 4.5l-1.73 1M3.67 10.5l-1.73 1M14.06 11.5l-1.73-1M3.67 5.5l-1.73-1M15 8h-2M3 8H1" />
    </>
  ),
  "chevron-right": <path d="M6 4l4 4-4 4" />,
  "chevron-down": <path d="M4 6l4 4 4-4" />,
  "chevron-up": <path d="M4 10l4-4 4 4" />,
  info: (
    <>
      <circle cx="8" cy="8" r="6" />
      <path d="M8 7.5v3.5M8 5v0.01" strokeLinecap="round" />
    </>
  ),
  warning: (
    <>
      <path d="M8 2.5L14 13H2L8 2.5z" />
      <path d="M8 7v3M8 11.5v0.01" />
    </>
  ),
  dot: <circle cx="8" cy="8" r="3" fill="currentColor" stroke="none" />,
  plus: (
    <>
      <path d="M8 3v10" />
      <path d="M3 8h10" />
    </>
  ),
  image: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="1" />
      <circle cx="6" cy="6.5" r="1" />
      <path d="M14 11l-3.5-3.5L4 14" />
    </>
  ),
  x: (
    <>
      <path d="M4 4l8 8" />
      <path d="M12 4l-8 8" />
    </>
  ),
  folder: (
    <>
      <path d="M2 5a1 1 0 0 1 1-1h3.5l1.5 1.5h5a1 1 0 0 1 1 1V12a1 1 0 0 1-1 1H3a1 1 0 0 1-1-1V5z" />
    </>
  ),
  lock: (
    <>
      <rect x="3.5" y="7" width="9" height="6.5" rx="1" />
      <path d="M5.5 7V5a2.5 2.5 0 0 1 5 0v2" />
    </>
  ),
  undo: (
    <>
      <path d="M6 4L2.5 7.5 6 11" />
      <path d="M3 7.5h6a4 4 0 0 1 4 4" />
    </>
  ),
  redo: (
    <>
      <path d="M10 4l3.5 3.5L10 11" />
      <path d="M13 7.5H7a4 4 0 0 0-4 4" />
    </>
  ),
  /* Restart glyph, not undo — discards state rather than navigating history. */
  refresh: (
    <>
      <path d="M13 8a5 5 0 1 1-1.6-3.7" />
      <path d="M13 2.5V5h-2.5" />
    </>
  ),
};
