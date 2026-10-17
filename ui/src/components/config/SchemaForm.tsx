import { useId, useState } from "react";
import type {
  ConfigObject,
  ConfigValue,
  FieldSchema,
  SectionSchema,
} from "@/lib/config/configSchema";
import { applyFieldChange, qualifyKey, readPath } from "@/lib/config/configSchema";
import {
  cm1FieldLandmark,
  cm1RelationshipLabel,
} from "@/lib/cm1/cm1ConfigCrosswalk";
import { Icon } from "@/components/shell/Icon";
import { clampNumberInput, validateNumberInput } from "@/lib/config/numberInput";
import "./SchemaForm.css";

interface SchemaFormProps {
  section: SectionSchema;
  config: ConfigObject;
  onChange: (next: ConfigObject) => void;
  showCompatibilityNames?: boolean;
}

/* Schema-driven form section. Visibility predicates run against the section subtree, not the root config. */
export function SchemaForm({ section, config, onChange, showCompatibilityNames = false }: SchemaFormProps) {
  const sectionRoot = readSectionRoot(config, section);
  const sectionLocked = section.locked === true;

  // Bool on/off switches owned by ModuleDisableControl are excluded here; scheme-based switches stay.
  const ownedBoolKey =
    section.disable &&
    section.fields.some(
      (f) =>
        qualifyKey(section, f) === section.disable!.path && f.kind === "bool",
    )
      ? section.disable.path
      : null;

  const visibleFields = section.fields.filter((f) => {
    if (ownedBoolKey && qualifyKey(section, f) === ownedBoolKey) return false;
    return !f.visibleWhen || f.visibleWhen(sectionRoot);
  });

  return (
    <div className="schema-form">
      {visibleFields.map((field) => {
        const fullKey = qualifyKey(section, field);
        const value = readPath(config, fullKey);
        const fieldDisabled = field.disabledWhen
          ? field.disabledWhen(config)
          : false;
        const fieldLocked = sectionLocked || field.locked === true;
        return (
          <FieldRow
            key={fullKey}
            field={field}
            nimbusPath={fullKey}
            value={value}
            disabled={fieldLocked || fieldDisabled}
            locked={fieldLocked}
            disabledReason={fieldDisabled && !fieldLocked ? field.disabledReason : undefined}
            showCompatibilityNames={showCompatibilityNames}
            onChange={(next) => onChange(applyFieldChange(config, fullKey, next))}
          />
        );
      })}
    </div>
  );
}

interface FieldRowProps {
  field: FieldSchema;
  nimbusPath?: string;
  value: ConfigValue | undefined;
  onChange: (next: ConfigValue) => void;
  disabled?: boolean;
  locked?: boolean;
  /** Inline disabled reason for conditionally disabled rows (locked rows use the padlock tooltip instead). */
  disabledReason?: string;
  showCompatibilityNames?: boolean;
  /** Explicit htmlFor/id pairing; label and control live in sibling subtrees. */
  controlId?: string;
}

function FieldRow({ field, nimbusPath, value, onChange, disabled, locked, disabledReason, showCompatibilityNames = false }: FieldRowProps) {
  const [showDescription, setShowDescription] = useState(false);
  const controlId = useId();
  const hasDescription = !!field.description;
  const cm1Landmark = nimbusPath ? cm1FieldLandmark(nimbusPath) : null;

  return (
    <div
      className={`schema-form__row ${disabled ? "schema-form__row--disabled" : ""} ${
        locked ? "schema-form__row--locked" : ""
      }`}
    >
      <div className="schema-form__label">
        {showCompatibilityNames && cm1Landmark && (
          <div
            className={`schema-form__cm1 schema-form__cm1--${cm1Landmark.relationship}`}
            title={cm1Landmark.note}
            role="note"
            aria-label={`CM1 ${cm1Landmark.cm1Keys.join(" or ")}: ${cm1Landmark.familiarConcept}. ${cm1Landmark.note}`}
          >
            <span>
              CM1 <code>{cm1Landmark.cm1Keys.join(" / ")}</code>
            </span>
            <span>{cm1RelationshipLabel(cm1Landmark.relationship)}</span>
          </div>
        )}
        <div className="schema-form__label-line">
          <label htmlFor={controlId}>{field.label}</label>
          {locked && (
            <span
              className="schema-form__lock"
              title={field.lockedReason ?? "Read-only: this value is derived, not chosen."}
            >
              <Icon name="lock" size={11} />
            </span>
          )}
          {hasDescription && (
            <button
              type="button"
              className={`schema-form__info-btn ${
                showDescription ? "schema-form__info-btn--active" : ""
              }`}
              aria-expanded={showDescription}
              aria-label={`More information about ${field.label}`}
              title={showDescription ? "Hide explanation" : "What does this do?"}
              onClick={() => setShowDescription((v) => !v)}
            >
              <Icon name="info" size={12} />
            </button>
          )}
        </div>
        {showCompatibilityNames && cm1Landmark && nimbusPath && (
          <span className="schema-form__canonical-id">
            Nimbus <code>{nimbusPath}</code>
          </span>
        )}
        {field.hint && <span className="schema-form__hint">{field.hint}</span>}
        {disabledReason && (
          <span className="schema-form__disabled-reason" role="note">
            <Icon name="lock" size={10} />
            <span>{disabledReason}</span>
          </span>
        )}
      </div>
      <div className="schema-form__control">
        <FieldControl
          field={field}
          value={value}
          onChange={onChange}
          disabled={disabled}
          controlId={controlId}
        />
      </div>
      {showDescription && field.description && (
        <div className="schema-form__description">{field.description}</div>
      )}
    </div>
  );
}

function FieldControl({ field, value, onChange, disabled, controlId }: FieldRowProps) {
  switch (field.kind) {
    case "int":
    case "float":
      return (
        <NumberControl
          field={field}
          value={value}
          onChange={onChange}
          disabled={disabled}
          controlId={controlId}
        />
      );
    case "bool":
      return (
        <BoolControl
          field={field}
          value={value}
          onChange={onChange}
          disabled={disabled}
          controlId={controlId}
        />
      );
    case "enum":
      return (
        <EnumControl
          field={field}
          value={value}
          onChange={onChange}
          disabled={disabled}
          controlId={controlId}
        />
      );
    case "string":
      return (
        <StringControl
          field={field}
          value={value}
          onChange={onChange}
          disabled={disabled}
          controlId={controlId}
        />
      );
  }
}

function NumberControl({ field, value, onChange, disabled, controlId }: FieldRowProps) {
  // Local draft of what the user is literally typing. A controlled number input
  // that refuses to commit "" snaps back to the previous value on clear AND
  // loses the selection, so the next keystrokes concatenate onto the old
  // number — clearing 128 and typing 96 used to yield 12896. The draft lets the
  // field sit in empty/partial states ("", "-", "0.") while editing; blur drops
  // it so the field resyncs to the last committed value.
  const [draft, setDraft] = useState<string | null>(null);
  if (field.kind !== "int" && field.kind !== "float") return null;
  const current =
    typeof value === "number" ? value : Number(field.default ?? 0);
  const isInt = field.kind === "int";
  const committed = Number.isFinite(current) ? String(current) : "";
  const validation = draft == null
    ? { valid: true, message: null, parsed: current }
    : validateNumberInput(draft, field.kind, field.min, field.max);
  const errorId = `${controlId}-range-error`;

  return (
    <div className="schema-form__number">
      <input
        id={controlId}
        type="number"
        value={draft ?? committed}
        step={field.step ?? (isInt ? 1 : "any")}
        min={field.min}
        max={field.max}
        disabled={disabled}
        aria-invalid={!validation.valid}
        aria-describedby={!validation.valid ? errorId : undefined}
        onChange={(e) => {
          const raw = e.target.value;
          setDraft(raw);
          if (raw === "") return;
          const next = validateNumberInput(raw, field.kind, field.min, field.max);
          if (next.valid && next.parsed != null) onChange(next.parsed);
        }}
        onBlur={() => {
          if (validation.parsed != null && !validation.valid) {
            onChange(clampNumberInput(validation.parsed, field.min, field.max));
          }
          setDraft(null);
        }}
      />
      {field.unit && <span className="schema-form__unit">{field.unit}</span>}
      {!validation.valid && <span id={errorId} className="schema-form__number-error" role="alert">{validation.message}</span>}
    </div>
  );
}

function BoolControl({
  field,
  value,
  onChange,
  disabled,
  controlId,
}: FieldRowProps) {
  if (field.kind !== "bool") return null;
  const on = value === true;
  return (
    <button
      id={controlId}
      type="button"
      className={`toggle ${on ? "toggle--on" : ""}`}
      disabled={disabled}
      onClick={() => onChange(!on)}
      aria-pressed={on}
      aria-label={field.label}
    >
      <span className="toggle__thumb" />
    </button>
  );
}

function EnumControl({ field, value, onChange, disabled, controlId }: FieldRowProps) {
  if (field.kind !== "enum") return null;
  const current =
    typeof value === "string" && field.options.some((o) => o.value === value)
      ? value
      : field.default;
  // Inline note for the currently selected option ("toy diagnostic only",
  // etc.) — surfaces caveats on enabled-but-imperfect schemes without
  // requiring the user to hover each dropdown entry to discover them.
  const activeNote = field.options.find((o) => o.value === current)?.note;
  const disabledReasons = field.showDisabledReasons
    ? field.options.filter(
      (option) => option.value !== current && option.disabled && option.note,
    )
    : [];
  return (
    <div className="enum-control">
      <select
        id={controlId}
        value={current}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value)}
      >
        {field.options.map((opt) => (
          <option
            key={opt.value}
            value={opt.value}
            disabled={opt.disabled}
            title={opt.note}
          >
            {opt.label}
            {opt.disabled ? " (unavailable)" : ""}
          </option>
        ))}
      </select>
      {activeNote && <span className="enum-control__note">{activeNote}</span>}
      {disabledReasons.map((option) => (
        <span className="enum-control__unavailable" key={`unavailable:${option.value}`}>
          <strong>{option.label}:</strong> {option.note}
        </span>
      ))}
    </div>
  );
}

function StringControl({ field, value, onChange, disabled, controlId }: FieldRowProps) {
  if (field.kind !== "string") return null;
  const current = typeof value === "string" ? value : "";
  return (
    <input
      id={controlId}
      type="text"
      value={current}
      placeholder={field.placeholder}
      disabled={disabled}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function readSectionRoot(
  config: ConfigObject,
  section: SectionSchema,
): ConfigObject {
  if (!section.path) return config;
  const sub = readPath(config, section.path);
  if (sub && typeof sub === "object" && !Array.isArray(sub)) {
    return sub as ConfigObject;
  }
  return {};
}
