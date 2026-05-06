"use client";

import type { FormEvent, RefObject } from "react";

export type SearchPanelMode = "bib" | "selfie";

export function BibPanel({
  bibInput,
  onBibChange,
  onSubmit,
  onSwitchToSelfie,
  photoCount,
  eventPhotoCount,
  inputRef,
}: {
  bibInput: string;
  onBibChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
  onSwitchToSelfie: () => void;
  photoCount: number;
  eventPhotoCount: number;
  inputRef?: RefObject<HTMLInputElement | null>;
}) {
  return (
    <form onSubmit={onSubmit} className="mt-8">
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mb-3">
        Your bib number
      </p>
      <label className="block">
        <span className="sr-only">Bib number</span>
        <input
          ref={inputRef}
          type="text"
          name="bib"
          inputMode="text"
          autoComplete="off"
          autoCapitalize="characters"
          value={bibInput}
          onChange={(e) => onBibChange(e.target.value)}
          placeholder="B-4082"
          className="block w-full border-b border-line bg-transparent focus:border-fresh outline-none font-mono tracking-[0.25em] text-lg py-3 placeholder:text-slate-soft text-ink"
        />
      </label>
      <button
        type="submit"
        className="mt-5 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors"
      >
        Search by bib →
      </button>

      <div className="mt-7 flex items-center gap-3">
        <span className="h-px flex-1 bg-line" />
        <span className="font-mono uppercase tracking-[0.3em] text-[9px] text-slate-soft">
          or
        </span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <button
        type="button"
        onClick={onSwitchToSelfie}
        className="mt-7 inline-flex items-center border border-ink hover:bg-ink hover:text-bone text-ink px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] transition-colors"
      >
        Upload a selfie →
      </button>

      <p className="mt-7 font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
        <span className="tnum">{photoCount.toLocaleString()}</span>{" "}
        {photoCount === 1 ? "photo" : "photos"}
        {eventPhotoCount > photoCount ? (
          <>
            {" "}
            of <span className="tnum">{eventPhotoCount.toLocaleString()}</span>
          </>
        ) : null}{" "}
        · free to search
      </p>
    </form>
  );
}

export function SelfiePendingPanel({
  onSwitchToBib,
}: {
  onSwitchToBib: () => void;
}) {
  return (
    <div className="mt-8" style={{ animation: "fade-in 0.3s ease-out both" }}>
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mb-3">
        Selfie match
      </p>
      <div className="rounded-xl border border-line bg-bone-deep px-5 py-6">
        <p className="font-display text-xl font-medium text-ink leading-snug">
          Coming soon.
        </p>
        <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
          We&apos;ll match your face against every photo in this race once the
          AI service is wired in. For now, search by bib.
        </p>
      </div>
      <button
        type="button"
        onClick={onSwitchToBib}
        className="mt-6 inline-flex items-center font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-fresh transition-colors"
      >
        ← Use bib instead
      </button>
    </div>
  );
}
