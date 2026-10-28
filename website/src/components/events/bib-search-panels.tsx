"use client";

import type { FormEvent, RefObject } from "react";
import { useEffect, useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useSelfiesList } from "@/hooks/use-selfies";
import { useQueryClient } from "@tanstack/react-query";
import { uploadSelfie } from "@/lib/api-selfies";
import { SELFIE_MAX } from "@/store/user-media-store";
import { useEffectiveRole } from "@/hooks/use-effective-role";
import { ROUTES } from "@/lib/constants";
import {
  searchEventByFace,
  type EventPhotosResult,
} from "@/lib/api-photos";
import { ApiError, searchRateLimitMessage } from "@/lib/api";
import { captureAuthSession } from "@/lib/auth-session";
import { FieldError } from "@/components/ui/field-error";
import { buildLoginRedirect } from "@/lib/redirect";
import {
  ACCEPTED_IMAGE_MIME,
  shrinkSelfie,
  validateImageFile,
} from "@/lib/image-utils";
import type { SelfieRef } from "@/store/user-media-store";
import { cn } from "@/lib/utils";

export type { SearchPanelMode } from "./find-photos-controls";
import { BibInput, SelfieActionCard, CameraGlyph, UploadGlyph } from "./find-photos-controls";

export function BibPanel({
  bibInput,
  onBibChange,
  onSubmit,
  onSwitchToSelfie,
  inputRef,
}: {
  bibInput: string;
  onBibChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
  onSwitchToSelfie: () => void;
  inputRef?: RefObject<HTMLInputElement | null>;
}) {
  return (
    <form onSubmit={onSubmit} className="mt-8">
      <BibInput bib={bibInput} onChange={onBibChange} inputRef={inputRef} />
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
  const sessionVersion = useAuthStore((s) => s.sessionVersion);
  const role = useEffectiveRole();
  const { selfies } = useSelfiesList();
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [libraryOpen, setLibraryOpen] = useState(false);
  // Save-and-search (2026-09-02): keep a fresh selfie in the library without
  // a detour to /profile. Default on; hidden when the library is full.
  const [saveToLibrary, setSaveToLibrary] = useState(true);
  const [notice, setNotice] = useState<string | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    setIsSearching(false);
    setError(null);
    setNotice(null);
    setLibraryOpen(false);
  }, [sessionVersion]);

  // Photographers don't have a runner selfie library — treat as guest for
  // the library affordance, but they still get upload/take in case they're
  // looking for themselves in another photographer's gallery.
  const isPhotographer = role === "PHOTOGRAPHER";
  const hasLibrary = isAuthenticated && !isPhotographer && selfies.length > 0;
  const canSave = isAuthenticated && !isPhotographer && selfies.length < SELFIE_MAX;
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
      return searchRateLimitMessage(err) ?? (err.message || "Could not match your face right now.");
    }
    return "Could not match your face right now. Try again.";
  }

  async function searchWithFile(file: File) {
    const isCurrent = captureAuthSession();
    const validation = validateImageFile(file);
    if (validation) {
      setError(validation);
      return;
    }
    setError(null);
    setNotice(null);
    setIsSearching(true);
    try {
      // Shrink once; the save and the fallback search send the same file.
      const upload = await shrinkSelfie(file);
      if (!isCurrent()) return;
      let result: EventPhotosResult | null = null;
      if (canSave && saveToLibrary) {
        // Save first, then match with the whole library (which now includes
        // it). A refused save (no face, limit) is a notice, not a failure —
        // the one-shot file search below still runs.
        try {
          await uploadSelfie(upload);
          if (!isCurrent()) return;
          void queryClient.invalidateQueries({ queryKey: ["me", "selfies"] });
          setNotice("Saved to your selfie library.");
          result = await searchEventByFace(eventSlug, { allSelfies: true });
        } catch (saveErr) {
          if (!isCurrent()) return;
          const why = saveErr instanceof ApiError ? saveErr.message : "upload failed";
          setNotice(`Not saved to your library (${why}) — searching with it anyway.`);
        }
      }
      if (!isCurrent()) return;
      if (!result) result = await searchEventByFace(eventSlug, { selfieFile: upload });
      if (!isCurrent()) return;
      onSearchSuccess(result);
    } catch (err) {
      if (!isCurrent()) return;
      setError(humanizeError(err));
    } finally {
      if (isCurrent()) setIsSearching(false);
    }
  }

  async function searchWithAllSelfies() {
    const isCurrent = captureAuthSession();
    setError(null);
    setIsSearching(true);
    try {
      const result = await searchEventByFace(eventSlug, { allSelfies: true });
      if (!isCurrent()) return;
      onSearchSuccess(result);
    } catch (err) {
      if (!isCurrent()) return;
      setError(humanizeError(err));
    } finally {
      if (isCurrent()) setIsSearching(false);
    }
  }

  async function searchWithSelfieId(selfieId: string) {
    const isCurrent = captureAuthSession();
    setError(null);
    setIsSearching(true);
    try {
      const result = await searchEventByFace(eventSlug, { selfieId });
      if (!isCurrent()) return;
      onSearchSuccess(result);
    } catch (err) {
      if (!isCurrent()) return;
      setError(humanizeError(err));
    } finally {
      if (isCurrent()) setIsSearching(false);
    }
  }

  return (
    <div className="mt-8" style={{ animation: "fade-in 0.3s ease-out both" }}>
      <p className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate mb-4">
        Selfie match · fastest way
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
            onMatchAll={() => {
              setLibraryOpen(false);
              void searchWithAllSelfies();
            }}
          />
        )}
      </div>

      {canSave && (
        <label className="mt-3 flex min-h-11 cursor-pointer items-center gap-3 font-sans text-sm text-slate select-none">
          <input
            type="checkbox"
            checked={saveToLibrary}
            onChange={(e) => setSaveToLibrary(e.target.checked)}
            disabled={isSearching}
            className="size-4 accent-fresh"
          />
          <span>
            Also save it to my selfie library{" "}
            <span className="font-mono tnum">({selfies.length} of {SELFIE_MAX})</span>
          </span>
        </label>
      )}

      {notice && (
        <p role="status" className="mt-2 font-sans text-sm text-slate">
          {notice}
        </p>
      )}

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
        className="mt-6 w-full inline-flex items-center justify-center rounded-xl border border-line px-6 py-4 font-mono uppercase tracking-[0.14em] text-[13px] text-ink hover:border-line-strong hover:bg-bone-deep transition-colors"
      >
        Search by bib →
      </button>
    </div>
  );
}

interface SelfieLibraryCardProps {
  isPrimary: boolean;
  disabled: boolean;
  open: boolean;
  onToggle: () => void;
  selfies: SelfieRef[];
  onPick: (selfie: SelfieRef) => void;
  /** Match with the whole library at once (backend unions the matches). */
  onMatchAll: () => void;
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
  onMatchAll,
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
          {/* Whole-library match first: every saved selfie is another angle of
              the same runner, and the backend unions the matches (2026-09-02).
              Ink, not Fresh — the card header above already carries the accent. */}
          <button
            type="button"
            onClick={onMatchAll}
            disabled={disabled}
            className="mb-3 w-full rounded-lg bg-ink px-4 py-3 text-left text-bone transition-colors hover:bg-ink/90 disabled:opacity-60 disabled:cursor-wait focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone-deep"
          >
            <span className="block font-display text-sm font-medium tracking-tight">
              Match with all <span className="font-mono tnum">{count}</span>{" "}
              {count === 1 ? "selfie" : "selfies"}
            </span>
            <span className="block font-sans text-sm text-bone/70 mt-0.5">
              Best results — different angles find more of your photos. Or pick one below.
            </span>
          </button>
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
