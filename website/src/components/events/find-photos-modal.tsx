"use client";

import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from "react";
import { createPortal } from "react-dom";
import { useScrollLock } from "@/lib/scroll-lock";
import {
  BibPanel,
  SelfieSearchPanel,
  type SearchPanelMode,
} from "./bib-search-panels";

interface FindPhotosModalProps {
  /** Event slug — passed to the selfie panel for the search-by-face POST. */
  eventSlug: string;
  /** Display name shown in the eyebrow above the headline. Usually `event.name`. */
  eyebrow: string;
  /** Number of photos in the current scope (event-wide or photographer-scoped). */
  photoCount: number;
  /** Total event photo count, for the "X of Y" caption. */
  eventPhotoCount: number;
  onClose: () => void;
  onSubmitBib: (b: string) => void;
  /** Fired after a successful face match. Defaults to closing the modal so
   *  callers without face-mode UI (e.g. /[handle]/events/[slug]) just dismiss. */
  onSearchByFaceSuccess?: () => void;
}

// Portal-mounted "Find your photos" modal. Reused by `event-cockpit` (event-wide
// search) and `[handle]/events/[slug]` (per-photographer search). Portaling
// matches the project rule that all overlays mount to `document.body` to
// escape ancestor stacking-context / containing-block traps.
export function FindPhotosModal({
  eventSlug,
  eyebrow,
  photoCount,
  eventPhotoCount,
  onClose,
  onSubmitBib,
  onSearchByFaceSuccess,
}: FindPhotosModalProps) {
  const [bibInput, setBibInput] = useState("");
  const [panelMode, setPanelMode] = useState<SearchPanelMode>("bib");
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const bibInputRef = useRef<HTMLInputElement | null>(null);
  const [mounted, setMounted] = useState(false);

  useScrollLock(true);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const focusables = () =>
      dialogRef.current
        ? Array.from(
            dialogRef.current.querySelectorAll<HTMLElement>(
              'button:not([tabindex="-1"]):not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
            ),
          ).filter((el) => !el.hasAttribute("disabled"))
        : [];

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab") {
        const items = focusables();
        if (items.length === 0) return;
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (e.shiftKey) {
          if (active === first || !dialogRef.current?.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last || !dialogRef.current?.contains(active)) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    document.addEventListener("keydown", onKey);

    return () => {
      document.removeEventListener("keydown", onKey);
      previouslyFocused?.focus();
    };
  }, [onClose]);

  useEffect(() => {
    if (panelMode === "bib") {
      bibInputRef.current?.focus();
    }
  }, [panelMode]);

  if (!mounted) return null;

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!bibInput.trim()) return;
    onClose();
    onSubmitBib(bibInput);
  };

  const content = (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label="Find your photos"
      className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6 md:p-10"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close search"
        tabIndex={-1}
        className="absolute inset-0 bg-ink/35 backdrop-blur-sm cursor-default"
        style={{ animation: "fade-in 0.2s ease-out both" }}
      />
      <div
        className="relative w-full max-w-md"
        style={{ animation: "fade-up 0.35s ease-out both" }}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close search"
          className="absolute -top-3 -right-3 z-10 size-9 rounded-full bg-ink text-bone flex items-center justify-center hover:bg-ink-soft transition-colors shadow-[0_8px_20px_-8px_rgba(17,17,17,0.45)] focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <svg
            viewBox="0 0 16 16"
            className="size-3.5"
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
        <div className="rounded-2xl bg-bone border border-line shadow-[0_24px_60px_-20px_rgba(17,17,17,0.45)] p-8 md:p-10 max-h-[85vh] overflow-y-auto">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-5">
            {eyebrow}
          </p>
          <h2 className="font-display text-4xl md:text-5xl font-medium tracking-tight leading-[0.95]">
            Find your
            <br />
            <span className="text-fresh">photos.</span>
          </h2>

          {panelMode === "bib" ? (
            <BibPanel
              bibInput={bibInput}
              onBibChange={setBibInput}
              onSubmit={handleSubmit}
              onSwitchToSelfie={() => setPanelMode("selfie")}
              photoCount={photoCount}
              eventPhotoCount={eventPhotoCount}
              inputRef={bibInputRef}
            />
          ) : (
            <SelfieSearchPanel
              eventSlug={eventSlug}
              onSwitchToBib={() => setPanelMode("bib")}
              onSearchSuccess={() => {
                if (onSearchByFaceSuccess) onSearchByFaceSuccess();
                onClose();
              }}
            />
          )}
        </div>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
