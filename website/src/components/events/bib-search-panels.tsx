"use client";

import type { FormEvent, RefObject } from "react";
import { useState } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useUserMediaStore } from "@/store/user-media-store";
import { ROUTES } from "@/lib/constants";
import { searchEventByFace } from "@/lib/api-photos";
import { ApiError } from "@/lib/api";
import { FieldError } from "@/components/ui/field-error";
import { buildLoginRedirect } from "@/lib/redirect";

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
        Match by selfie →
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

interface SelfieSearchPanelProps {
  eventSlug: string;
  onSwitchToBib: () => void;
  onSearchSuccess: () => void;
}

// Renders one of three states depending on auth + selfie-library status:
//   guest                 → "Sign in to use selfie search"
//   authed, no primary    → "Add a selfie first" → /profile#selfies?next=...
//   authed, has primary   → "Match my face" button (POSTs to search-by-face)
//
// Replaces the legacy `<SelfiePendingPanel>` placeholder. Kept as a named
// export so the cockpit can `import { SelfiePendingPanel }` without breaking
// (alias points at the new component).
export function SelfieSearchPanel({
  eventSlug,
  onSwitchToBib,
  onSearchSuccess,
}: SelfieSearchPanelProps) {
  const router = useRouter();
  const pathname = usePathname();
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const role = useAuthStore((s) => s.user?.role);
  const primarySelfie = useUserMediaStore((s) =>
    s.selfies.find((sf) => sf.isPrimary),
  );
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const here = pathname || `/events/${eventSlug}`;

  const handleSignIn = () => {
    router.push(buildLoginRedirect(here));
  };

  const handleAddSelfie = () => {
    router.push(`${ROUTES.PROFILE}?next=${encodeURIComponent(here)}#selfies`);
  };

  const handleMatch = async () => {
    if (!primarySelfie) return;
    setIsSearching(true);
    setError(null);
    try {
      await searchEventByFace(eventSlug, { selfieId: primarySelfie.id });
      onSearchSuccess();
    } catch (err) {
      if (err instanceof ApiError && err.errors[0]?.code === "LOW_CONFIDENCE") {
        setError(
          "We didn't find your face in this event yet. Try another selfie.",
        );
      } else {
        setError(
          err instanceof ApiError
            ? err.message
            : "Could not match your face right now. Try again.",
        );
      }
    } finally {
      setIsSearching(false);
    }
  };

  // Photographers don't have a selfie library — the surface is runner-only.
  // Show the same "sign in / add a selfie" copy as a guest.
  const treatAsGuest = !isAuthenticated || role === "PHOTOGRAPHER";

  return (
    <div className="mt-8" style={{ animation: "fade-in 0.3s ease-out both" }}>
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mb-3">
        Selfie match
      </p>

      {treatAsGuest ? (
        <div className="rounded-xl border border-line bg-bone-deep px-5 py-6">
          <p className="font-display text-xl font-medium text-ink leading-snug">
            Sign in to match.
          </p>
          <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
            We use your saved selfie to find every photo of you in this race.
          </p>
          <button
            type="button"
            onClick={handleSignIn}
            className="mt-4 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-5 py-2.5 rounded-full font-mono uppercase tracking-[0.2em] text-[12px] transition-colors"
          >
            Sign in →
          </button>
        </div>
      ) : !primarySelfie ? (
        <div className="rounded-xl border border-line bg-bone-deep px-5 py-6">
          <p className="font-display text-xl font-medium text-ink leading-snug">
            Add a selfie first.
          </p>
          <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
            We&apos;ll match it against every photo in this race once it&apos;s on
            file. Two minutes in your profile.
          </p>
          <button
            type="button"
            onClick={handleAddSelfie}
            className="mt-4 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-5 py-2.5 rounded-full font-mono uppercase tracking-[0.2em] text-[12px] transition-colors"
          >
            Open profile →
          </button>
        </div>
      ) : (
        <div className="rounded-xl border border-line bg-bone-deep px-5 py-6">
          <p className="font-display text-xl font-medium text-ink leading-snug">
            Match my face.
          </p>
          <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
            We&apos;ll scan this race for your saved selfie. Free to look — pay
            only for keepers.
          </p>
          <button
            type="button"
            onClick={handleMatch}
            disabled={isSearching}
            className="mt-4 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-5 py-2.5 rounded-full font-mono uppercase tracking-[0.2em] text-[12px] transition-colors disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {isSearching ? "Matching…" : "Match my face →"}
          </button>
          <FieldError
            message={error}
            id="selfie-search-error"
            density="tight"
          />
        </div>
      )}

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

// Backwards-compat alias — `<SelfiePendingPanel>` is the historical name.
export const SelfiePendingPanel = SelfieSearchPanel;
