"use client";

import { type ChangeEvent, type ReactNode } from "react";
import { FieldError } from "@/components/ui/field-error";
import { cn } from "@/lib/utils";

// Phase 6 admin form-field primitives. Every form-bearing destructive
// modal in /admin composes from this kit so label / input / counter /
// hint styling can never drift between modals. Quiet Studio canonical:
// mono uppercase labels at the readability floor, rounded-2xl line input
// on bg-surface with fresh focus ring, mono tnum counter under each
// length-capped field, warning-text hint when input is invalid.

const LABEL_CLS =
  "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft block";
const COUNTER_CLS =
  "font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft text-right tnum";
const HINT_CLS =
  "font-mono uppercase tracking-[0.18em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-warning-text";
const INPUT_BASE =
  "w-full rounded-2xl border border-line bg-surface py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh";
const INPUT_PADDING_DEFAULT = "px-4";
const INPUT_PADDING_PREFIX = "pl-9 pr-4";

export function AdminFieldLabel({
  htmlFor,
  required,
  children,
}: {
  htmlFor?: string;
  required?: boolean;
  children: ReactNode;
}) {
  return htmlFor ? (
    <label htmlFor={htmlFor} className={LABEL_CLS}>
      {children}
      {required && <span className="text-slate"> (required)</span>}
    </label>
  ) : (
    <p className={LABEL_CLS}>{children}</p>
  );
}

export function AdminFieldCounter({ children }: { children: ReactNode }) {
  return <p className={COUNTER_CLS}>{children}</p>;
}

export function AdminFieldHint({ id, children }: { id?: string; children: ReactNode }) {
  return <p id={id} className={HINT_CLS}>{children}</p>;
}

interface FormSectionProps {
  label: string;
  children: ReactNode;
}
// A fieldset whose legend names the cluster (radio groups, and any field
// cluster where the label can't be a proper `<label htmlFor>`).
export function AdminFormSection({ label, children }: FormSectionProps) {
  return (
    // No m-0: it would beat the parent's space-y gap; Tailwind's reset already zeroes fieldset margin, border and padding.
    <fieldset className="min-w-0 space-y-3">
      <legend className={LABEL_CLS}>{label}</legend>
      {children}
    </fieldset>
  );
}

interface RadioOption {
  value: string;
  label: ReactNode;
}
interface RadioGroupProps {
  name: string;
  options: ReadonlyArray<RadioOption>;
  value: string;
  onChange: (next: string) => void;
}
export function AdminRadioGroup({
  name,
  options,
  value,
  onChange,
}: RadioGroupProps) {
  return (
    <div>
      {options.map((opt) => (
        // 44 px rows; first/last negative margins merge with the parent's gap (a -my-3 on the list would replace it).
        <label
          key={opt.value}
          className="flex min-h-[44px] items-start gap-3 py-3 first:-mt-3 last:-mb-3 cursor-pointer"
        >
          <input
            type="radio"
            name={name}
            value={opt.value}
            checked={value === opt.value}
            onChange={() => onChange(opt.value)}
            className="mt-1 accent-ink"
          />
          <span className="font-sans text-sm text-ink-soft">{opt.label}</span>
        </label>
      ))}
    </div>
  );
}

interface TextInputProps {
  id: string;
  label: ReactNode;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  maxLength?: number;
  /** Single-character / short-string prefix shown inside the input on the left (e.g. "@"). */
  prefix?: string;
  /** When true, render an `n/maxLength` counter under the input. Requires maxLength. */
  showCounter?: boolean;
  type?: "text" | "number";
  min?: number | string;
  max?: number | string;
  step?: number | string;
  inputMode?: "text" | "numeric" | "decimal" | "none";
  autoFocus?: boolean;
  /** Optional warning-text hint message rendered under the input. */
  hint?: ReactNode;
  /** Extra classes for the input element itself (e.g. "tnum" on numeric inputs). */
  inputClassName?: string;
  /**
   * Optional sanitizer applied before slicing to maxLength — used by the
   * edit-photographer handle field which strips non-handle chars.
   */
  sanitize?: (raw: string) => string;
  /** Inline error; pass it only after the field was left or a submit was tried. */
  error?: string | null;
  required?: boolean;
}
export function AdminTextInput({
  id,
  label,
  value,
  onChange,
  placeholder,
  maxLength,
  prefix,
  showCounter = false,
  type = "text",
  min,
  max,
  step,
  inputMode,
  autoFocus,
  hint,
  inputClassName,
  sanitize,
  error,
  required,
}: TextInputProps) {
  const hintId = hint ? `${id}-hint` : undefined;
  const errorId = error ? `${id}-error` : undefined;
  const a11y = {
    required,
    "aria-invalid": error ? true : undefined,
    "aria-describedby": [hintId, errorId].filter(Boolean).join(" ") || undefined,
  };
  const handle = (e: ChangeEvent<HTMLInputElement>) => {
    let next = e.target.value;
    if (sanitize) next = sanitize(next);
    if (maxLength != null) next = next.slice(0, maxLength);
    onChange(next);
  };
  const inputCls = cn(
    INPUT_BASE,
    prefix ? INPUT_PADDING_PREFIX : INPUT_PADDING_DEFAULT,
    inputClassName,
  );
  return (
    <div className="space-y-2">
      <AdminFieldLabel htmlFor={id} required={required}>{label}</AdminFieldLabel>
      {prefix ? (
        <div className="relative">
          <span
            aria-hidden
            className="absolute left-4 top-1/2 -translate-y-1/2 font-mono text-sm text-slate-soft"
          >
            {prefix}
          </span>
          <input
            id={id}
            type={type}
            value={value}
            onChange={handle}
            placeholder={placeholder}
            min={min}
            max={max}
            step={step}
            inputMode={inputMode}
            autoFocus={autoFocus}
            className={inputCls}
            {...a11y}
          />
        </div>
      ) : (
        <input
          id={id}
          type={type}
          value={value}
          onChange={handle}
          placeholder={placeholder}
          min={min}
          max={max}
          step={step}
          inputMode={inputMode}
          autoFocus={autoFocus}
          className={inputCls}
          {...a11y}
        />
      )}
      {hint && <AdminFieldHint id={hintId}>{hint}</AdminFieldHint>}
      <FieldError id={`${id}-error`} message={error} density="tight" />
      {showCounter && maxLength != null && (
        <AdminFieldCounter>
          {value.length}/{maxLength}
        </AdminFieldCounter>
      )}
    </div>
  );
}

interface TextareaProps {
  id: string;
  label: ReactNode;
  value: string;
  onChange: (next: string) => void;
  placeholder?: string;
  maxLength: number;
  rows?: number;
  /** Defaults to true — every length-capped textarea in admin shows its counter. */
  showCounter?: boolean;
  /** Inline error; pass it only after the field was left or a submit was tried. */
  error?: string | null;
  required?: boolean;
  /** Id of a plain hint shown outside the field, read out with it. */
  describedBy?: string;
  disabled?: boolean;
}
export function AdminTextarea({
  id,
  label,
  value,
  onChange,
  placeholder,
  maxLength,
  rows = 3,
  showCounter = true,
  error,
  required,
  describedBy,
  disabled,
}: TextareaProps) {
  return (
    <div className="space-y-2">
      <AdminFieldLabel htmlFor={id} required={required}>{label}</AdminFieldLabel>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, maxLength))}
        rows={rows}
        placeholder={placeholder}
        className={cn(INPUT_BASE, INPUT_PADDING_DEFAULT, "resize-none disabled:opacity-50 disabled:cursor-not-allowed")}
        required={required}
        disabled={disabled}
        aria-invalid={error ? true : undefined}
        aria-describedby={[describedBy, error && `${id}-error`].filter(Boolean).join(" ") || undefined}
      />
      <FieldError id={`${id}-error`} message={error} density="tight" />
      {showCounter && (
        <AdminFieldCounter>
          {value.length}/{maxLength}
        </AdminFieldCounter>
      )}
    </div>
  );
}
