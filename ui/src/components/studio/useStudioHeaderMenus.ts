import { useLayoutEffect, type RefObject } from "react";

export function useStudioHeaderMenus(header: RefObject<HTMLElement>, sourceIdentity: string | null) {
  useLayoutEffect(() => {
    const host = header.current;
    const studio = host?.closest<HTMLElement>(".plot-studio");
    if (!host || !studio) return;
    const menus = () => [...host.querySelectorAll<HTMLDetailsElement>(".plot-studio__compare-menu")];
    const position = () => {
      const bounds = studio.getBoundingClientRect();
      studio.style.setProperty("--studio-header-bottom", `${host.getBoundingClientRect().bottom - bounds.top}px`);
      const left = Math.max(0, bounds.left) + 8;
      const right = Math.min(window.innerWidth, bounds.right) - 8;
      const bottom = Math.min(window.innerHeight, bounds.bottom) - 8;
      for (const menu of menus().filter(menu => menu.open)) {
        const popup = menu.querySelector<HTMLElement>(".plot-studio__compare-menu-popover");
        if (!popup) continue;
        popup.style.maxWidth = `${Math.max(0, right - left)}px`;
        popup.style.transform = "none";
        const rect = popup.getBoundingClientRect();
        const shift = Math.max(left - rect.left, Math.min(0, right - rect.right));
        popup.style.transform = `translateX(${shift}px)`;
        popup.style.maxHeight = `${Math.max(0, bottom - rect.top)}px`;
      }
    };
    const toggle = (event: Event) => {
      if (!(event.target instanceof HTMLDetailsElement) || !event.target.open) return;
      for (const menu of menus()) if (menu !== event.target) menu.open = false;
      position();
    };
    const dismissOutside = (event: Event) => {
      if (!(event.target instanceof Node)) return;
      for (const menu of menus()) if (menu.open && !menu.contains(event.target)) menu.open = false;
    };
    const escape = (event: KeyboardEvent) => {
      const open = menus().find(menu => menu.open);
      if (event.key !== "Escape" || !open) return;
      event.preventDefault();
      event.stopPropagation();
      open.open = false;
      open.querySelector<HTMLElement>("summary")?.focus();
    };
    host.addEventListener("toggle", toggle, true);
    document.addEventListener("pointerdown", dismissOutside);
    document.addEventListener("focusin", dismissOutside);
    document.addEventListener("keydown", escape, true);
    window.addEventListener("resize", position);
    const observer = new ResizeObserver(position);
    observer.observe(host);
    observer.observe(studio);
    position();
    return () => {
      observer.disconnect();
      host.removeEventListener("toggle", toggle, true);
      document.removeEventListener("pointerdown", dismissOutside);
      document.removeEventListener("focusin", dismissOutside);
      document.removeEventListener("keydown", escape, true);
      window.removeEventListener("resize", position);
    };
  }, [header, sourceIdentity]);
}
