"use client";

import { type ChangeEvent, type ReactNode } from "react";
import { cn } from "@/lib/utils";

// Phase 6 admin form-field primitives. Every form-bearing destructive
// modal in /admin composes from this kit so label / input / counter /
// hint styling can never drift between modals. Quiet Studio canonical:
// mono uppercase labels at the readability floor, rounded-2xl line input
// on bg-surface with fresh focus ring, mono tnum counter under each
// length-capped field, amber-700 hint when input is invalid.

const LABEL_CLS =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft block";
const COUNTER_CLS =
  "font-mono uppercase tracking-[0.18em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft text-right tnum";
const HINT_CLS =
  "font-mono uppercase tracking-[0.18em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-amber-700";
const INPUT_BASE =
  "w-full rounded-2xl border border-line bg-surface py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh";
const INPUT_PADDING_DEFAULT = "px-4";
const INPUT_PADDING_PREFIX = "pl-9 pr-4";

export function AdminFieldLabel({
  htmlFor,
  children,
}: {
  htmlFor?: string;
  children: ReactNode;
}) {
  return htmlFor ? (
    <label htmlFor={htmlFor} className={LABEL_CLS}>
      {children}
    </label>
  ) : (
    <p className={LABEL_CLS}>{children}</p>
  );
}

export function AdminFieldCounter({ children }: { children: ReactNode }) {
  return <p className={COUNTER_CLS}>{children}</p>;
}

export function AdminFieldHint({ children }: { children: ReactNode }) {
  return <p className={HINT_CLS}>{children}</p>;
}

interface FormSectionProps {
  label: string;
  children: ReactNode;
}
// Section with a mono label heading above its content. Used for radio
// groups (and any field cluster where the label can't be a proper
// `<label htmlFor>`).
export function AdminFormSection({ label, children }: FormSectionProps) {
  return (
    <div className="space-y-3">
      <AdminFieldLabel>{label}</AdminFieldLabel>
      {children}
    </div>
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
    <div className="space-y-2">
      {options.map((opt) => (
        <label
          key={opt.value}
          className="flex items-start gap-3 cursor-pointer"
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
  /** Optional amber-700 hint message rendered under the input. */
  hint?: ReactNode;
  /** Extra classes for the input element itself (e.g. "tnum" on numeric inputs). */
  inputClassName?: string;
  /**
   * Optional sanitizer applied before slicing to maxLength — used by the
   * edit-photographer handle field which strips non-handle chars.
   */
  sanitize?: (raw: string) => string;
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
}: TextInputProps) {
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
      <AdminFieldLabel htmlFor={id}>{label}</AdminFieldLabel>
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
        />
      )}
      {hint && <AdminFieldHint>{hint}</AdminFieldHint>}
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
}: TextareaProps) {
  return (
    <div className="space-y-2">
      <AdminFieldLabel htmlFor={id}>{label}</AdminFieldLabel>
      <textarea
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, maxLength))}
        rows={rows}
        placeholder={placeholder}
        className={cn(INPUT_BASE, INPUT_PADDING_DEFAULT, "resize-none")}
      />
      {showCounter && (
        <AdminFieldCounter>
          {value.length}/{maxLength}
        </AdminFieldCounter>
      )}
    </div>
  );
}
