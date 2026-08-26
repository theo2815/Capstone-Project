"use client";

import type { ChangeEvent, FormEvent, RefObject } from "react";
import { useRef, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useSelfiesList } from "@/hooks/use-selfies";
import { useEffectiveRole } from "@/hooks/use-effective-role";
import { ROUTES } from "@/lib/constants";
import {
  searchEventByFace,
  type EventPhotosResult,
} from "@/lib/api-photos";
import { ApiError } from "@/lib/api";
import { FieldError } from "@/components/ui/field-error";
import { buildLoginRedirect } from "@/lib/redirect";
import {
  ACCEPTED_IMAGE_MIME,
  validateImageFile,
} from "@/lib/image-utils";
import type { SelfieRef } from "@/store/user-media-store";
import { cn } from "@/lib/utils";

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
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate mb-3">
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
          className="block w-full border-b border-line bg-transparent focus:border-fresh outline-none font-mono tracking-[0.14em] text-lg py-3 placeholder:text-slate-soft text-ink"
        />
      </label>
      <button
        type="submit"
        className="mt-5 inline-flex items-center bg-fresh hover:bg-fresh-deep text-surface px-6 py-3 rounded-full font-display font-bold text-[15px] transition-colors"
      >
        Search by bib →
      </button>

      <div className="mt-7 flex items-center gap-3">
        <span className="h-px flex-1 bg-line" />
        <span className="font-mono uppercase tracking-[0.14em] text-[9px] text-slate-soft">
          or
        </span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <button
        type="button"
        onClick={onSwitchToSelfie}
        className="mt-7 inline-flex items-center border border-ink hover:bg-ink hover:text-bone text-ink px-6 py-3 rounded-full font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors"
      >
        Match by selfie →
      </button>

      <p className="mt-7 font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft">
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

interface SelfieSearchPanelProps {
  eventSlug: string;
  onSwitchToBib: () => void;
  onSearchSuccess: (result: EventPhotosResult) => void;
}

// Three explicit affordances for runner-side face search (2026-05-19 PM
// redesign — removes the implicit "Match by primary selfie" CTA):
//   · Take a selfie     → file input with capture="user" (mobile opens
//                         native camera; desktop falls back to picker)
//   · Upload a selfie   → standard file picker
//   · Use saved selfie  → only when the authed runner has at least one
//                         stored selfie; expands inline thumbnail grid;
//                         clicking a thumb fires the selfieId-based search.
//
// Guests + photographers see Take + Upload only (no library). The library
// path is the only one that needs auth; multipart upload+take work for
// anyone — backend `searchByFaceMultipart` is public-readable.
//
// All paths await the search and pass EventPhotosResult up via onSearchSuccess
// so the cockpit can render photos directly — face search is now one-shot,
// no client-side caching by selfieId.
export function SelfieSearchPanel({
  eventSlug,
  onSwitchToBib,
  onSearchSuccess,
}: SelfieSearchPanelProps) {
  const router = useRouter();
  const pathname = usePathname();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const role = useEffectiveRole();
  const { selfies } = useSelfiesList();
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);

  // Photographers don't have a runner selfie library — treat as guest for
  // the library affordance, but they still get upload/take in case they're
  // looking for themselves in another photographer's gallery.
  const isPhotographer = role === "PHOTOGRAPHER";
  const hasLibrary = isAuthenticated && !isPhotographer && selfies.length > 0;
  const here = pathname || `/events/${eventSlug}`;

  function humanizeError(err: unknown): string {
    if (err instanceof ApiError) {
      const code = err.errors[0]?.code;
      if (code === "LOW_CONFIDENCE") {
        return "We didn't find your face in this event yet. Try another shot.";
      }
      if (code === "SELFIE_REJECTED") {
        return err.message || "Selfie rejected — try a clearer, frontal shot.";
      }
      if (code === "AI_API_UNAVAILABLE") {
        return "Face search is offline right now. Try again in a few minutes.";
      }
      return err.message || "Could not match your face right now.";
    }
    return "Could not match your face right now. Try again.";
  }

  async function searchWithFile(file: File) {
    const validation = validateImageFile(file);
    if (validation) {
      setError(validation);
      return;
    }
    setError(null);
    setIsSearching(true);
    try {
      const result = await searchEventByFace(eventSlug, { selfieFile: file });
      onSearchSuccess(result);
    } catch (err) {
      setError(humanizeError(err));
    } finally {
      setIsSearching(false);
    }
  }

  async function searchWithSelfieId(selfieId: string) {
    setError(null);
    setIsSearching(true);
    try {
      const result = await searchEventByFace(eventSlug, { selfieId });
      onSearchSuccess(result);
    } catch (err) {
      setError(humanizeError(err));
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <div className="mt-8" style={{ animation: "fade-in 0.3s ease-out both" }}>
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate mb-4">
        Selfie match
      </p>

      <div className="space-y-3">
        <SelfieActionCard
          label="Take a selfie"
          caption="Use your camera to snap one now."
          isPrimary={!hasLibrary}
          disabled={isSearching}
          accept={ACCEPTED_IMAGE_MIME.join(",")}
          captureUser
          onFile={(f) => void searchWithFile(f)}
          glyph={<CameraGlyph />}
        />
        <SelfieActionCard
          label="Upload a selfie"
          caption="Pick an existing photo from your device."
          isPrimary={false}
          disabled={isSearching}
          accept={ACCEPTED_IMAGE_MIME.join(",")}
          captureUser={false}
          onFile={(f) => void searchWithFile(f)}
          glyph={<UploadGlyph />}
        />
        {hasLibrary && (
          <SelfieLibraryCard
            isPrimary={true}
            disabled={isSearching}
            open={libraryOpen}
            onToggle={() => setLibraryOpen((v) => !v)}
            selfies={selfies}
            onPick={(s) => {
              setLibraryOpen(false);
              void searchWithSelfieId(s.id);
            }}
          />
        )}
      </div>

      <FieldError
        message={error}
        id="selfie-search-error"
        density="tight"
      />

      {isSearching && (
        <p
          role="status"
          className="mt-4 font-mono uppercase tracking-[0.14em] text-[10px] text-fresh"
        >
          Matching…
        </p>
      )}

      {!isAuthenticated && (
        <p className="mt-6 font-sans text-sm text-slate leading-relaxed">
          Want to save selfies for next time?{" "}
          <button
            type="button"
            onClick={() => router.push(buildLoginRedirect(here))}
            className="underline decoration-line underline-offset-4 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            Sign in →
          </button>
        </p>
      )}

      {isAuthenticated && !isPhotographer && selfies.length === 0 && (
        <p className="mt-6 font-sans text-sm text-slate leading-relaxed">
          Tip: build a{" "}
          <button
            type="button"
            onClick={() =>
              router.push(`${ROUTES.PROFILE}?next=${encodeURIComponent(here)}#selfies`)
            }
            className="underline decoration-line underline-offset-4 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            selfie library
          </button>{" "}
          and reuse it across every race.
        </p>
      )}

      <button
        type="button"
        onClick={onSwitchToBib}
        className="mt-6 inline-flex items-center font-mono uppercase tracking-[0.14em] text-[10px] text-slate hover:text-fresh transition-colors"
      >
        ← Use bib instead
      </button>
    </div>
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
function SelfieActionCard({
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
        disabled && "opacity-60 cursor-wait",
      )}
    >
      <span
        aria-hidden="true"
        className={cn(
          "size-9 shrink-0 grid place-items-center rounded-full",
          isPrimary
            ? "bg-bone/15 text-bone"
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
            isPrimary ? "text-bone/75" : "text-slate",
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

interface SelfieLibraryCardProps {
  isPrimary: boolean;
  disabled: boolean;
  open: boolean;
  onToggle: () => void;
  selfies: SelfieRef[];
  onPick: (selfie: SelfieRef) => void;
}

// Library card collapses into the same action-row shape as Take/Upload when
// closed, and expands into a thumbnail grid when open. Clicking a thumb fires
// the search immediately (no extra "confirm" step) — primary is marked so
// runners can spot their default at a glance.
function SelfieLibraryCard({
  isPrimary,
  disabled,
  open,
  onToggle,
  selfies,
  onPick,
}: SelfieLibraryCardProps) {
  const count = selfies.length;
  return (
    <div>
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-expanded={open}
        className={cn(
          "w-full flex items-center gap-4 rounded-xl px-5 py-4 transition-colors text-left",
          isPrimary
            ? "bg-fresh text-surface hover:bg-fresh-deep"
            : "border border-line bg-bone hover:bg-bone-deep text-ink",
          disabled && "opacity-60 cursor-wait",
        )}
      >
        <span
          aria-hidden="true"
          className={cn(
            "size-9 shrink-0 grid place-items-center rounded-full",
            isPrimary ? "bg-bone/15 text-bone" : "border border-line text-slate",
          )}
        >
          <LibraryGlyph />
        </span>
        <span className="flex-1 min-w-0">
          <span className="block font-display text-base font-medium tracking-tight leading-snug">
            Use saved selfie
          </span>
          <span
            className={cn(
              "block font-sans text-sm leading-snug mt-0.5",
              isPrimary ? "text-bone/75" : "text-slate",
            )}
          >
            <span className="font-mono tnum">{count}</span>{" "}
            {count === 1 ? "selfie" : "selfies"} in your library.
          </span>
        </span>
        <span
          aria-hidden="true"
          className={cn(
            "font-mono text-base transition-transform",
            open && "rotate-90",
          )}
        >
          →
        </span>
      </button>

      {open && (
        <div
          className="mt-3 rounded-xl border border-line bg-bone-deep p-3"
          style={{ animation: "fade-in 0.2s ease-out both" }}
        >
          <ul className="grid grid-cols-3 gap-2.5 sm:grid-cols-4">
            {selfies.map((s) => (
              <li key={s.id}>
                <button
                  type="button"
                  onClick={() => onPick(s)}
                  disabled={disabled}
                  aria-label={
                    s.isPrimary ? `Search with primary selfie` : `Search with this selfie`
                  }
                  className="group relative aspect-square w-full overflow-hidden rounded-lg bg-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep disabled:opacity-60 disabled:cursor-wait"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element -- selfie URLs are signed S3 outside Next image-domain config. */}
                  <img
                    src={s.dataUrl}
                    alt=""
                    className="size-full object-cover transition-transform group-hover:scale-105"
                    draggable={false}
                  />
                  {s.isPrimary && (
                    <span className="absolute top-1.5 left-1.5 font-mono uppercase tracking-[0.18em] text-[8px] bg-fresh text-surface px-1.5 py-0.5 rounded-full">
                      Primary
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function CameraGlyph() {
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

function UploadGlyph() {
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

function LibraryGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="size-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="2" y="3" width="5" height="5" rx="1" />
      <rect x="9" y="3" width="5" height="5" rx="1" />
      <rect x="2" y="10" width="5" height="3.5" rx="1" />
      <rect x="9" y="10" width="5" height="3.5" rx="1" />
    </svg>
  );
}

// Backwards-compat alias — `<SelfiePendingPanel>` is the historical name.
export const SelfiePendingPanel = SelfieSearchPanel;
