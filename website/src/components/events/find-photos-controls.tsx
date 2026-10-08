"use client";

import { useId, useRef, type ChangeEvent, type RefObject } from "react";
import { FieldError } from "@/components/ui/field-error";
import { BTN_PRIMARY, BTN_GHOST, BTN_SIZE } from "@/components/ui/button-styles";
import { ACCEPTED_IMAGE_MIME } from "@/lib/image-utils";
import { cn } from "@/lib/utils";

export type SearchPanelMode = "bib" | "selfie";

export function BibInput({ bib, onChange, inputRef, disabled = false }: {
  bib: string; onChange: (value: string) => void; inputRef?: RefObject<HTMLInputElement | null>; disabled?: boolean;
}) {
  const inputId = useId();
  return (
      <div className="relative flex items-center rounded-xl border border-line bg-transparent focus-within:border-fresh transition-colors">
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          name="bib"
          inputMode="text"
          autoComplete="off"
          autoCapitalize="characters"
          value={bib}
          onChange={(e) => onChange(e.target.value)}
          placeholder=" "
          disabled={disabled}
          className="peer block w-full bg-transparent outline-none font-mono tnum tracking-[0.14em] text-lg px-5 pt-6 pb-2 text-ink"
        />
        <label
          htmlFor={inputId}
          className="pointer-events-none absolute left-5 font-sans text-slate-soft transition-all duration-150 top-1/2 -translate-y-1/2 text-lg peer-focus:top-2 peer-focus:translate-y-0 peer-focus:text-sm peer-focus:text-slate peer-[:not(:placeholder-shown)]:top-2 peer-[:not(:placeholder-shown)]:translate-y-0 peer-[:not(:placeholder-shown)]:text-sm peer-[:not(:placeholder-shown)]:text-slate"
        >
          Enter your bib number
        </label>
        {bib && (
          <button
            type="button"
            onClick={() => {
              onChange("");
              inputRef?.current?.focus();
            }}
            aria-label="Clear bib number"
            disabled={disabled}
            className="shrink-0 pl-2 pr-5 text-slate hover:text-ink transition-colors"
          >
            <svg
              viewBox="0 0 16 16"
              className="size-4"
              fill="none"
              aria-hidden="true"
            >
              <path
                d="M3 3 L13 13 M13 3 L3 13"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
              />
            </svg>
          </button>
        )}
      </div>
  );
}

export interface FindPhotosControlsProps {
  file: File | null;
  bib: string;
  onFileChange: (file: File | null) => void;
  onBibChange: (bib: string) => void;
  onSearch: () => void;
  busy: boolean;
  error: string | null;
  allowPersonalLibrary: boolean;
}

// Public adapters own the library; case controls only report controlled input.
export function FindPhotosControls({ file, bib, onFileChange, onBibChange, onSearch, busy, error, allowPersonalLibrary }: FindPhotosControlsProps) {
  const errorId = useId();
  return (
    <form onSubmit={(event) => { event.preventDefault(); if (!busy) onSearch(); }} className="space-y-4">
      <SelfieActionCard label="Take a selfie" caption="Use your camera to snap one now." isPrimary={false} disabled={busy} accept={ACCEPTED_IMAGE_MIME.join(",")} captureUser onFile={onFileChange} glyph={<CameraGlyph />} />
      <SelfieActionCard label="Upload a selfie" caption="Pick an existing photo from your device." isPrimary={false} disabled={busy} accept={ACCEPTED_IMAGE_MIME.join(",")} captureUser={false} onFile={onFileChange} glyph={<UploadGlyph />} />
      {file && <div className="flex min-w-0 items-center gap-3"><span className="min-w-0 truncate text-sm text-ink">{file.name}</span><button type="button" aria-label="Remove selected selfie" disabled={busy} onClick={() => onFileChange(null)} className={cn(BTN_GHOST, BTN_SIZE.sm)}>Remove</button></div>}
      {!allowPersonalLibrary && <p className="text-sm text-slate">This selfie is private case evidence.</p>}
      <BibInput bib={bib} onChange={onBibChange} disabled={busy} />
      <FieldError message={error} id={errorId} density="tight" />
      <button type="submit" disabled={busy || (!file && !bib.trim())} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>{busy ? "Searching…" : "Search"}</button>
    </form>
  );
}

interface SelfieActionCardProps {
  label: string;
  caption: string;
  isPrimary: boolean;
  disabled: boolean;
  accept: string;
  captureUser: boolean;
  onFile: (file: File) => void;
  glyph: React.ReactNode;
}

// One-shot file input wrapped in a Quiet Studio card. capture="user" hints
// the browser to open the front camera on mobile; on desktop the attribute
// is silently ignored and the standard picker appears. Either way the result
// flows through the same handleFile path.
export function SelfieActionCard({
  label,
  caption,
  isPrimary,
  disabled,
  accept,
  captureUser,
  onFile,
  glyph,
}: SelfieActionCardProps) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (file) onFile(file);
  };
  return (
    <label
      className={cn(
        "group flex items-center gap-4 rounded-xl px-5 py-4 transition-colors",
        isPrimary
          ? "bg-fresh text-surface hover:bg-fresh-deep cursor-pointer"
          : "border border-line bg-bone hover:bg-bone-deep text-ink cursor-pointer",
        disabled && "opacity-50 cursor-not-allowed",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "size-9 shrink-0 grid place-items-center rounded-full",
          isPrimary
            ? "bg-surface/15 text-surface"
            : "border border-line text-slate group-hover:text-ink",
        )}
      >
        {glyph}
      </span>
      <span className="flex-1 min-w-0">
        <span className="block font-display text-base font-medium tracking-tight leading-snug">
          {label}
        </span>
        <span
          className={cn(
            "block font-sans text-sm leading-snug mt-0.5",
            isPrimary ? "text-surface/75" : "text-slate",
          )}
        >
          {caption}
        </span>
      </span>
      <span aria-hidden="true" className="font-mono text-base">
        →
      </span>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        {...(captureUser ? { capture: "user" } : {})}
        onChange={handleChange}
        disabled={disabled}
        className="sr-only"
      />
    </label>
  );
}

export function CameraGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="size-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinejoin="round"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path d="M2 5h2l1.2-1.7h5.6L12 5h2v8H2V5z" />
      <circle cx="8" cy="9" r="2.4" />
    </svg>
  );
}

export function UploadGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="size-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M8 11V2.5" />
      <path d="M5 5.5L8 2.5L11 5.5" />
      <path d="M2.5 11v2A1.5 1.5 0 0 0 4 14.5h8a1.5 1.5 0 0 0 1.5-1.5v-2" />
    </svg>
  );
}
