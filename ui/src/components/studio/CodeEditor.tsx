import { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import CodeMirror, { type ReactCodeMirrorRef } from "@uiw/react-codemirror";
import { python } from "@codemirror/lang-python";
import { oneDark } from "@codemirror/theme-one-dark";
import {
  Decoration,
  EditorView,
  keymap,
  ViewPlugin,
  type DecorationSet,
  type ViewUpdate,
} from "@codemirror/view";
import { autocompletion, type CompletionSource } from "@codemirror/autocomplete";
import { syntaxTree } from "@codemirror/language";
import { Prec, RangeSetBuilder } from "@codemirror/state";
import { isCellMarkerLine } from "@/lib/studio/pythonCells";
import {
  candidateIsVisible,
  parseCompletionPrefix,
  type CompletionCandidate,
} from "@/lib/studio/pythonCompletion";
import {
  currentStudioAppearance,
  STUDIO_APPEARANCE_APPLIED_EVENT,
  type ResolvedStudioAppearance,
} from "@/lib/studio/studioAppearance";
import "./CodeEditor.css";

/* CodeMirror 6 wrapper. Handlers read through refs so extensions stay stable despite per-keystroke callback changes. */
export interface CodeEditorHandle {
  /** Insert a snippet at the current cursor / replace the active
   *  selection. Caret lands at end of inserted text. */
  insertAtCursor: (snippet: string) => void;
  /** Move keyboard focus into the editor. */
  focus: () => void;
  /** Character offset of the caret, or 0 when the editor isn't mounted. */
  getCursorOffset: () => number;
  /** Put the caret at `offset` and scroll it into view. */
  setCursorOffset: (offset: number) => void;
}

interface CodeEditorProps {
  id?: string;
  ariaLabelledBy?: string;
  ariaLabel?: string;
  value: string;
  onChange: (next: string) => void;
  /** Cmd/Ctrl+Enter — run, without moving the caret. */
  onSubmit?: () => void;
  /** Shift+Enter — run and advance to the next cell. */
  onSubmitAdvance?: () => void;
  /** Cmd/Ctrl+Shift+Enter — run the whole script. */
  onRunAll?: () => void;
  /** Cmd/Ctrl+S; also suppresses the browser Save-Page dialog. */
  onSave?: () => void;
  /** Caret moved, so the parent can track which cell is current. */
  onCursorChange?: (offset: number) => void;
  /** Name completions for a dotted path + partial prefix; empty list disables the popup. */
  completeNames?: (
    path: readonly string[],
    partial: string,
  ) => Promise<readonly CompletionCandidate[]>;
  /** Extra className on the outer wrapper (set to flex:1 / min-height:0 so it fills its container). */
  className?: string;
  placeholder?: string;
}

// Line decoration (not widget) to avoid shifting gutter line numbers away from real file numbering.
const cellMarkerDecoration = Decoration.line({ class: "cm-nimbus-cell-marker" });

function buildCellDecorations(view: EditorView): DecorationSet {
  const builder = new RangeSetBuilder<Decoration>();
  // Only the visible ranges: a 5000-line script would otherwise be scanned in
  // full on every viewport change.
  for (const { from, to } of view.visibleRanges) {
    let pos = from;
    while (pos <= to) {
      const line = view.state.doc.lineAt(pos);
      if (isCellMarkerLine(line.text)) {
        builder.add(line.from, line.from, cellMarkerDecoration);
      }
      pos = line.to + 1;
    }
  }
  return builder.finish();
}

const cellMarkerHighlight = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;

    constructor(view: EditorView) {
      this.decorations = buildCellDecorations(view);
    }

    update(update: ViewUpdate) {
      if (update.docChanged || update.viewportChanged) {
        this.decorations = buildCellDecorations(update.view);
      }
    }
  },
  { decorations: (plugin) => plugin.decorations },
);

// No completions inside strings, comments, or cell markers (prose, not code).
const NON_CODE_NODES = new Set(["Comment", "String", "FormatString", "TripleString"]);

function pythonCompletionSource(
  lookup: (path: readonly string[], partial: string) => Promise<readonly CompletionCandidate[]>,
): CompletionSource {
  return async (context) => {
    const node = syntaxTree(context.state).resolveInner(context.pos, -1);
    if (NON_CODE_NODES.has(node.name)) return null;

    const line = context.state.doc.lineAt(context.pos);
    const request = parseCompletionPrefix(line.text.slice(0, context.pos - line.from));
    if (!request) return null;
    // No prefix: only offer after a dot, unless `explicit` (user pressed the completion key directly).
    if (!context.explicit && request.partial.length === 0 && request.path.length === 0) {
      return null;
    }

    const answered = await lookup(request.path, request.partial);
    // Stale result guard: document may have changed while awaiting; CM sets `aborted`.
    if (context.aborted) return null;

    const candidates = answered
      .filter((candidate) => candidateIsVisible(candidate.label, request.partial));
    if (candidates.length === 0) return null;

    return {
      from: context.pos - request.backtrack,
      options: candidates.map((candidate) => ({
        label: candidate.label,
        type: candidate.kind,
        detail: candidate.detail ?? undefined,
      })),
      // CM narrows this result locally while the user keeps typing the same word.
      validFor: /^[A-Za-z0-9_]*$/,
    };
  };
}

export const CodeEditor = forwardRef<CodeEditorHandle, CodeEditorProps>(
  function CodeEditor(
    {
      value,
      id,
      ariaLabelledBy,
      ariaLabel,
      onChange,
      onSubmit,
      onSubmitAdvance,
      onRunAll,
      onSave,
      onCursorChange,
      completeNames,
      className,
      placeholder,
    },
    ref,
  ) {
    const cmRef = useRef<ReactCodeMirrorRef>(null);
    const [appearance, setAppearance] = useState<ResolvedStudioAppearance>(() => currentStudioAppearance());

    const handlers = useRef({
      onSubmit,
      onSubmitAdvance,
      onRunAll,
      onSave,
      onCursorChange,
      completeNames,
    });
    useEffect(() => {
      handlers.current = {
        onSubmit,
        onSubmitAdvance,
        onRunAll,
        onSave,
        onCursorChange,
        completeNames,
      };
    }, [onSubmit, onSubmitAdvance, onRunAll, onSave, onCursorChange, completeNames]);

    useEffect(() => {
      const onAppearance = () => setAppearance(currentStudioAppearance());
      window.addEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
      return () => window.removeEventListener(STUDIO_APPEARANCE_APPLIED_EVENT, onAppearance);
    }, []);

    const insertAtCursor = useCallback((snippet: string) => {
      const view = cmRef.current?.view;
      if (!view) return;
      const { from, to } = view.state.selection.main;
      view.dispatch({
        changes: { from, to, insert: snippet },
        selection: { anchor: from + snippet.length },
      });
      view.focus();
    }, []);

    const focus = useCallback(() => {
      cmRef.current?.view?.focus();
    }, []);

    const getCursorOffset = useCallback(() => {
      return cmRef.current?.view?.state.selection.main.head ?? 0;
    }, []);

    const setCursorOffset = useCallback((offset: number) => {
      const view = cmRef.current?.view;
      if (!view) return;
      const clamped = Math.max(0, Math.min(offset, view.state.doc.length));
      view.dispatch({
        selection: { anchor: clamped },
        scrollIntoView: true,
      });
      view.focus();
    }, []);

    useImperativeHandle(ref, () => ({ insertAtCursor, focus, getCursorOffset, setCursorOffset }), [
      insertAtCursor,
      focus,
      getCursorOffset,
      setCursorOffset,
    ]);

    // No deps: configured once; handlers via ref. Bindings return false when no handler is wired (CM default applies).
    const extensions = useMemo(
      () => [
        python(),
        EditorView.lineWrapping,
        cellMarkerHighlight,
        // Built once via ref; `validFor` lets CM narrow locally per keystroke after the first dot lookup.
        autocompletion({
          override: [
            pythonCompletionSource(async (path, partial) =>
              await handlers.current.completeNames?.(path, partial) ?? [],
            ),
          ],
          icons: false,
          closeOnBlur: true,
        }),
        EditorView.updateListener.of((update: ViewUpdate) => {
          if (update.selectionSet || update.docChanged) {
            handlers.current.onCursorChange?.(update.state.selection.main.head);
          }
        }),
        Prec.highest(
          keymap.of([
            {
              key: "Mod-Enter",
              run: () => {
                if (!handlers.current.onSubmit) return false;
                handlers.current.onSubmit();
                return true;
              },
            },
            {
              key: "Shift-Enter",
              run: () => {
                if (!handlers.current.onSubmitAdvance) return false;
                handlers.current.onSubmitAdvance();
                return true;
              },
            },
            {
              key: "Mod-Shift-Enter",
              run: () => {
                if (!handlers.current.onRunAll) return false;
                handlers.current.onRunAll();
                return true;
              },
            },
            {
              key: "Mod-s",
              run: () => {
                if (!handlers.current.onSave) return false;
                handlers.current.onSave();
                return true;
              },
            },
          ]),
        ),
      ],
      [],
    );

    return (
      <div
        id={id}
        className={`code-editor ${className ?? ""}`}
        role="tabpanel"
        aria-labelledby={ariaLabelledBy}
      >
        <CodeMirror
          ref={cmRef}
          value={value}
          onChange={onChange}
          theme={appearance === "dark" ? oneDark : "light"}
          extensions={extensions}
          placeholder={placeholder}
          aria-label={ariaLabel}
          basicSetup={{
            lineNumbers: true,
            highlightActiveLine: true,
            foldGutter: false,
            searchKeymap: true,
          }}
          height="100%"
          style={{ height: "100%" }}
        />
      </div>
    );
  },
);
