import { useId, useState, type ReactNode } from "react";
import { Icon } from "@/components/shell/Icon";
import type { ThreeDControlScopeState } from "@/lib/plot/threeDControlScope";

export interface Grouped3DControlSection {
  id: string;
  label: string;
  status: string;
  controls: ReactNode;
  ariaLabel?: string;
}

export function Grouped3DControls({
  groups,
  ariaLabel,
  scope,
  notice,
}: {
  groups: readonly Grouped3DControlSection[];
  ariaLabel: string;
  scope: ThreeDControlScopeState;
  notice?: string | null;
}) {
  const [openGroup, setOpenGroup] = useState<string | null>(null);
  const instanceId = useId().replaceAll(":", "");
  const active = groups.find((group) => group.id === openGroup) ?? null;

  return (
    <div className="volume-render__controls volume-render__controls--grouped">
      <div className="plot3d-control-toolbar" role="toolbar" aria-label={ariaLabel}>
        {groups.map((group) => {
          const open = group.id === openGroup;
          const panelId = `plot3d-controls-${instanceId}-${group.id}`;
          return (
            <button
              key={group.id}
              type="button"
              className={`plot3d-control-tab${open ? " plot3d-control-tab--active" : ""}`}
              data-state={group.status.toLowerCase()}
              aria-expanded={open}
              title={`${group.label}: ${group.status}`}
              aria-controls={panelId}
              onClick={() => setOpenGroup((current) => current === group.id ? null : group.id)}
            >
              <span>{group.label}</span>
              <span className="plot3d-control-tab__status">{group.status}</span>
              <Icon name={open ? "chevron-up" : "chevron-down"} size={12} />
            </button>
          );
        })}
        <span
          className={`plot3d-control-notice${notice ? " is-visible" : ""}`}
          role="status"
          aria-hidden={notice ? undefined : true}
        >
          {notice ?? "Preview while moving · settles to full quality"}
        </span>
        <ThreeDControlScopeToggle state={scope} />
      </div>
      {active && (
        <div
          id={`plot3d-controls-${instanceId}-${active.id}`}
          className="plot3d-control-panel"
          role="region"
          aria-label={active.ariaLabel ?? `${active.label} controls`}
        >
          {active.controls}
        </div>
      )}
    </div>
  );
}

export function ThreeDControlScopeToggle({ state }: { state: ThreeDControlScopeState }) {
  return (
    <div className="plot3d-control-scope" role="group" aria-label="Apply common 3D settings to">
      <span className="plot3d-control-scope__label">Apply to</span>
      <button
        type="button"
        className={state.scope === "independent" ? "plot3d-control-scope__option is-active" : "plot3d-control-scope__option"}
        aria-pressed={state.scope === "independent"}
        title="Background, lighting, scene scale, and radar floor affect only this 3D view"
        onClick={() => state.setScope("independent")}
      >
        This view
      </button>
      <button
        type="button"
        className={state.scope === "shared" ? "plot3d-control-scope__option is-active" : "plot3d-control-scope__option"}
        aria-pressed={state.scope === "shared"}
        title="Background, lighting, scene scale, and radar floor stay synchronized across 3D views"
        onClick={() => state.setScope("shared")}
      >
        All 3D
      </button>
    </div>
  );
}
