import type { PageId } from "@/lib/core/types";
import { ariaShortcut, commandForPage, isMacPlatform, shortcutLabel } from "@/lib/studio/commandRegistry";
import { Icon, type IconName } from "./Icon";
import "./NavRail.css";

interface NavRailProps {
  activePage: PageId;
  onSelect: (page: PageId) => void;
  onOpenCommandPalette: () => void;
}

interface NavItem {
  id: PageId;
  label: string;
  icon: IconName;
}

const NAV_ITEMS: NavItem[] = [
  { id: "projects", label: "Projects", icon: "folder" },
  { id: "config", label: "Configure", icon: "play" },
  { id: "runs", label: "Runs", icon: "archive" },
  { id: "plots", label: "Plots", icon: "plot" },
  { id: "settings", label: "Settings", icon: "settings" },
];

export function NavRail({ activePage, onSelect, onOpenCommandPalette }: NavRailProps) {
  const mac = isMacPlatform(globalThis.navigator?.platform ?? "");
  return (
    <nav className="nav-rail" aria-label="Primary navigation">
      <div className="nav-rail__brand" aria-label="Nimbus Studio">
        <BrandMark />
      </div>
      <ul className="nav-rail__list">
        {NAV_ITEMS.map((item) => {
          const isActive = item.id === activePage;
          const command = commandForPage(item.id);
          return (
            <li key={item.id}>
              <button
                type="button"
                className={`nav-rail__item ${isActive ? "nav-rail__item--active" : ""}`}
                onClick={() => onSelect(item.id)}
                aria-current={isActive ? "page" : undefined}
                title={`${item.label} (${shortcutLabel(command, mac)})`}
                aria-keyshortcuts={ariaShortcut(command, mac)}
              >
                <Icon name={item.icon} size={18} />
                <span className="nav-rail__label">{item.label}</span>
              </button>
            </li>
          );
        })}
      </ul>
      <button
        type="button"
        className="nav-rail__commands"
        onClick={onOpenCommandPalette}
        aria-label={`Open command palette (${mac ? "Command K" : "Control K"})`}
        aria-keyshortcuts={mac ? "Meta+K" : "Control+K"}
        title={`Command palette (${mac ? "⌘K" : "Ctrl+K"})`}
      >
        <span aria-hidden="true">K</span>
        <span className="nav-rail__label">Commands</span>
      </button>
    </nav>
  );
}

/* Compact atmospheric contour mark; restrained enough for an instrument rail. */
function BrandMark() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" aria-hidden="true">
      <rect width="28" height="28" rx="5" fill="var(--color-accent)" />
      <path
        d="M9 19V9l10 10V9"
        stroke="var(--color-accent-contrast)"
        strokeWidth="2"
        fill="none"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
