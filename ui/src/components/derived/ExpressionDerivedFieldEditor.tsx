import { useMemo, useState } from "react";
import {
  MAX_EXPRESSION_DERIVED_FIELDS,
  createExpressionDerivedField,
  type ExpressionDerivedFieldDefinition,
  type ExpressionDerivedFieldDraft,
  type ExpressionDerivedFieldParameter,
} from "@/lib/derived/expressionDerivedFields";
import "./ExpressionDerivedFieldEditor.css";

export interface ExpressionDerivedFieldEditorProps {
  datasetLabel: string;
  definitions: readonly ExpressionDerivedFieldDefinition[];
  nativeFields: ReadonlySet<string>;
  reservedFieldIds?: ReadonlySet<string>;
  availableFields?: ReadonlySet<string>;
  onDefinitionsChange: (definitions: readonly ExpressionDerivedFieldDefinition[]) => void | Promise<void>;
}

interface DraftText {
  id: string;
  label: string;
  unit: string;
  dependencies: string;
  parameters: string;
  formula: string;
}

const EMPTY_DRAFT: DraftText = {
  id: "",
  label: "",
  unit: "",
  dependencies: "",
  parameters: "",
  formula: "",
};

function definitionToDraft(definition: ExpressionDerivedFieldDefinition): DraftText {
  return {
    id: definition.id,
    label: definition.label,
    unit: definition.unit,
    dependencies: definition.dependencies.join(", "),
    parameters: definition.parameters.map(({ id, value }) => `${id}=${value}`).join(", "),
    formula: definition.formula,
  };
}

function parseList(value: string): string[] {
  return value.split(",").map((entry) => entry.trim()).filter(Boolean);
}

function parseParameters(value: string): ExpressionDerivedFieldParameter[] {
  if (!value.trim()) return [];
  return parseList(value).map((entry) => {
    const separator = entry.indexOf("=");
    if (separator < 1) throw new Error(`parameter '${entry}' must use name=value`);
    const id = entry.slice(0, separator).trim();
    const raw = entry.slice(separator + 1).trim();
    if (!raw) throw new Error(`parameter '${id}' is missing a value`);
    return { id, value: Number(raw) };
  });
}

function draftContract(draft: DraftText): ExpressionDerivedFieldDraft {
  return {
    id: draft.id,
    label: draft.label,
    unit: draft.unit,
    dependencies: parseList(draft.dependencies),
    parameters: parseParameters(draft.parameters),
    formula: draft.formula,
  };
}

export function ExpressionDerivedFieldEditor({
  datasetLabel,
  definitions,
  nativeFields,
  reservedFieldIds,
  availableFields = nativeFields,
  onDefinitionsChange,
}: ExpressionDerivedFieldEditorProps) {
  const [draft, setDraft] = useState<DraftText>(EMPTY_DRAFT);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const existingIds = useMemo(() => new Set(definitions.map(({ id }) => id)), [definitions]);
  const validation = useMemo(() => {
    try {
      return {
        definition: createExpressionDerivedField(draftContract(draft), {
          nativeFields,
          reservedFieldIds,
          availableFields,
          existingDefinitionIds: existingIds,
          replacingId: editingId ?? undefined,
        }),
        error: null,
      };
    } catch (error) {
      return { definition: null, error: error instanceof Error ? error.message : String(error) };
    }
  }, [availableFields, draft, editingId, existingIds, nativeFields, reservedFieldIds]);

  const reset = () => {
    setDraft(EMPTY_DRAFT);
    setEditingId(null);
    setSubmitError(null);
  };

  const submit = async () => {
    if (!validation.definition || busy) return;
    setBusy(true);
    setSubmitError(null);
    try {
      const next = editingId
        ? definitions.map((definition) => definition.id === editingId ? validation.definition! : definition)
        : [...definitions, validation.definition];
      await onDefinitionsChange([...next].sort((left, right) => left.id.localeCompare(right.id)));
      reset();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (id: string) => {
    if (busy) return;
    setBusy(true);
    setSubmitError(null);
    try {
      await onDefinitionsChange(definitions.filter((definition) => definition.id !== id));
      if (editingId === id) reset();
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const atLimit = definitions.length >= MAX_EXPRESSION_DERIVED_FIELDS && editingId == null;

  return <section className="expression-fields" aria-labelledby="expression-fields-title">
    <header className="expression-fields__header">
      <div>
        <span className="expression-fields__eyebrow">Dataset-local registry</span>
        <h3 id="expression-fields-title">Expression fields</h3>
        <p>{datasetLabel} · {definitions.length}/{MAX_EXPRESSION_DERIVED_FIELDS} definitions</p>
      </div>
      <span className="expression-fields__version">schema v1</span>
    </header>

    <div className="expression-fields__instrument">
      <div className="expression-fields__form">
        <label>Field ID<input value={draft.id} disabled={editingId != null} placeholder="cold_pool_speed" onChange={(event) => setDraft({ ...draft, id: event.target.value })} /></label>
        <label>Display label<input value={draft.label} placeholder="Cold-pool speed" onChange={(event) => setDraft({ ...draft, label: event.target.value })} /></label>
        <label className="expression-fields__unit">Output unit<input value={draft.unit} placeholder="m/s" onChange={(event) => setDraft({ ...draft, unit: event.target.value })} /></label>
        <label className="expression-fields__wide">Dependencies<input value={draft.dependencies} placeholder="u, v" onChange={(event) => setDraft({ ...draft, dependencies: event.target.value })} /><small>Comma-separated, same-frame fields. Every declaration must be used.</small></label>
        <label className="expression-fields__wide">Formula<textarea rows={3} spellCheck={false} value={draft.formula} placeholder="sqrt(u^2 + v^2)" onChange={(event) => setDraft({ ...draft, formula: event.target.value })} /></label>
        <label className="expression-fields__wide">Fixed parameters <span>optional</span><input value={draft.parameters} placeholder="threshold=10, scale=0.001" onChange={(event) => setDraft({ ...draft, parameters: event.target.value })} /><small>Finite numeric name=value pairs become part of field and cache identity.</small></label>
        <div className={`expression-fields__validation ${validation.definition ? "expression-fields__validation--ready" : ""}`} role="status">
          <strong>{validation.definition ? "Definition qualified" : "Definition incomplete"}</strong>
          <span>{validation.definition ? `${validation.definition.dependencies.length} exact dependencies · output [${validation.definition.unit}]` : validation.error}</span>
        </div>
        {submitError && <p className="expression-fields__error" role="alert">{submitError}</p>}
        <div className="expression-fields__actions">
          <button type="button" className="expression-fields__commit" disabled={!validation.definition || atLimit || busy} onClick={submit}>{busy ? "Applying…" : editingId ? "Update definition" : "Add definition"}</button>
          {(editingId || draft.id || draft.formula) && <button type="button" className="expression-fields__quiet" disabled={busy} onClick={reset}>Clear</button>}
        </div>
      </div>

      <aside className="expression-fields__boundary" aria-label="Expression execution boundary">
        <strong>Execution boundary</strong>
        <dl>
          <div><dt>Scope</dt><dd>One cell, one frame</dd></div>
          <div><dt>Invalids</dt><dd>Propagate as NaN</dd></div>
          <div><dt>Maximum</dt><dd>5,000,000 cells</dd></div>
          <div><dt>Compute</dt><dd>Cancellable worker</dd></div>
          <div><dt>Collision</dt><dd>Native output wins</dd></div>
        </dl>
        <p>Evaluation runs off the interface thread and an obsolete request terminates its worker. No cross-frame references, resampling, implicit unit conversion, spatial filling, or arbitrary JavaScript.</p>
      </aside>
    </div>

    <div className="expression-fields__registry" aria-label="Saved expression fields">
      {definitions.length === 0 ? <p className="expression-fields__empty">No user expressions are registered for this dataset. Begin with a canonical field ID and declare every input explicitly.</p> : definitions.map((definition) => <article key={definition.id} className="expression-fields__definition">
        <div><strong>{definition.label}</strong><code>{definition.id} [{definition.unit}]</code></div>
        <code className="expression-fields__formula">{definition.formula}</code>
        <span>{definition.dependencies.join(" + ")}{definition.parameters.length ? ` · ${definition.parameters.length} fixed` : ""}</span>
        <div className="expression-fields__row-actions">
          <button type="button" disabled={busy} onClick={() => { setDraft(definitionToDraft(definition)); setEditingId(definition.id); setSubmitError(null); }}>Edit</button>
          <button type="button" disabled={busy} onClick={() => void remove(definition.id)}>Delete</button>
        </div>
      </article>)}
    </div>
  </section>;
}
