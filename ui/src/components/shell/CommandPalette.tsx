import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  STUDIO_COMMANDS,
  filterCommands,
  isEditableTarget,
  isMacPlatform,
  matchesCommandShortcut,
  shortcutLabel,
  type StudioCommand,
} from "@/lib/studio/commandRegistry";
import "./CommandPalette.css";

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onExecute: (command: StudioCommand) => void;
}

export function CommandPalette({ open, onOpenChange, onExecute }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(0);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const openRef = useRef(open);
  openRef.current = open;
  const mac = useMemo(() => isMacPlatform(globalThis.navigator?.platform ?? ""), []);
  const commands = useMemo(() => filterCommands(STUDIO_COMMANDS, query), [query]);

  const close = useCallback(() => {
    onOpenChange(false);
    setQuery("");
    setActiveIndex(0);
  }, [onOpenChange]);

  const execute = useCallback((command: StudioCommand) => {
    close();
    onExecute(command);
  }, [close, onExecute]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      returnFocusRef.current = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
      dialog.showModal();
      requestAnimationFrame(() => inputRef.current?.focus());
    } else if (!open && dialog.open) {
      dialog.close();
      // WebKit can leave `activeElement` on the now-hidden search field after
      // Escape. Besides being an accessibility defect, that makes the next
      // global command look as though it was typed into an editor. Restore the
      // opener synchronously; blur as a safe fallback if it has disappeared.
      const returnTarget = returnFocusRef.current;
      if (returnTarget?.isConnected) returnTarget.focus();
      else if (document.activeElement instanceof HTMLElement) document.activeElement.blur();
      returnFocusRef.current = null;
    }
  }, [open]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const currentlyOpen = openRef.current;
      const paletteShortcut = (mac ? event.metaKey : event.ctrlKey)
        && !event.altKey && !event.shiftKey && event.key.toLocaleLowerCase() === "k";
      if (paletteShortcut && (currentlyOpen || !isEditableTarget(event.target))) {
        event.preventDefault();
        onOpenChange(!currentlyOpen);
        return;
      }
      if (currentlyOpen || isEditableTarget(event.target)) return;
      const command = STUDIO_COMMANDS.find((candidate) => matchesCommandShortcut(event, candidate, mac));
      if (!command) return;
      event.preventDefault();
      execute(command);
    };
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [execute, mac, onOpenChange]);

  useEffect(() => setActiveIndex(0), [query]);

  return createPortal(
    <dialog
      ref={dialogRef}
      className="command-palette"
      aria-labelledby="command-palette-title"
      onCancel={(event) => { event.preventDefault(); close(); }}
      onClose={close}
      onClick={(event) => { if (event.target === dialogRef.current) close(); }}
    >
      <div className="command-palette__instrument">
        <header className="command-palette__header">
          <div>
            <span className="command-palette__eyebrow">NIMBUS STUDIO</span>
            <h2 id="command-palette-title">Command palette</h2>
          </div>
          <kbd>{mac ? "⌘K" : "Ctrl+K"}</kbd>
        </header>
        <label className="command-palette__search" htmlFor="command-palette-search">
          <span className="sr-only">Find a command</span>
          <input
            ref={inputRef}
            id="command-palette-search"
            type="search"
            role="combobox"
            aria-autocomplete="list"
            aria-expanded="true"
            aria-controls="command-palette-results"
            aria-activedescendant={commands[activeIndex] ? `command-${commands[activeIndex].id}` : undefined}
            value={query}
            placeholder="Navigate or find a safe action"
            autoComplete="off"
            onChange={(event) => setQuery(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setActiveIndex((index) => commands.length ? (index + 1) % commands.length : 0);
              } else if (event.key === "ArrowUp") {
                event.preventDefault();
                setActiveIndex((index) => commands.length ? (index - 1 + commands.length) % commands.length : 0);
              } else if (event.key === "Enter" && commands[activeIndex]) {
                event.preventDefault();
                execute(commands[activeIndex]);
              }
            }}
          />
        </label>
        <div id="command-palette-results" className="command-palette__results" role="listbox" aria-label="Available commands">
          {commands.length ? commands.map((command, index) => (
            <button
              key={command.id}
              id={`command-${command.id}`}
              type="button"
              role="option"
              aria-selected={index === activeIndex}
              className={index === activeIndex ? "command-palette__command command-palette__command--active" : "command-palette__command"}
              onMouseEnter={() => setActiveIndex(index)}
              onClick={() => execute(command)}
            >
              <span className="command-palette__command-copy">
                <strong>{command.label}</strong>
                <span>{command.description}</span>
              </span>
              <span className="command-palette__command-meta">
                <span>{command.group}</span>
                {command.shortcut && <kbd>{shortcutLabel(command, mac)}</kbd>}
              </span>
            </button>
          )) : (
            <div className="command-palette__empty" role="status">
              <strong>No matching command</strong>
              <span>Try a page name, “netCDF,” or “run.”</span>
            </div>
          )}
        </div>
        <footer className="command-palette__footer">
          <span><kbd>↑</kbd><kbd>↓</kbd> choose</span>
          <span><kbd>Enter</kbd> open</span>
          <span><kbd>Esc</kbd> close</span>
          <span>Launch, abort, and delete are intentionally unavailable here.</span>
        </footer>
      </div>
    </dialog>,
    document.body,
  );
}
