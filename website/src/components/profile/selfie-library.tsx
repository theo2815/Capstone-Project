"use client";

import { useState, type ChangeEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  SELFIE_MAX,
  type SelfieRef,
} from "@/store/user-media-store";
import { useSelfiesList } from "@/hooks/use-selfies";
import { useToast } from "@/hooks/use-toast";
import { Kicker } from "@/components/ui/kicker";
import { ApiError } from "@/lib/api";
import {
  uploadSelfie,
  deleteSelfie,
  setPrimarySelfie,
} from "@/lib/api-selfies";
import {
  ACCEPTED_IMAGE_MIME,
  validateImageFile,
} from "@/lib/image-utils";

export function SelfieLibrary() {
  const { selfies } = useSelfiesList();
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const remaining = SELFIE_MAX - selfies.length;
  const primary = selfies.find((s) => s.isPrimary);

  function invalidateSelfieDependents() {
    queryClient.invalidateQueries({ queryKey: ["me", "selfies"] });
    // No event-photo invalidation here: face-search results are one-shot
    // component state (never cached under a query key), and the
    // ["events",slug,"photos",{bib}] caches hold bib/browse results that
    // don't depend on the selfie set — the old predicate nuke only refetched
    // innocent caches.
  }

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    e.target.value = "";
    if (files.length === 0) return;

    setBusy(true);
    setError(null);
    try {
      const accepted = files.slice(0, remaining);
      let firstSkipReason: string | null = null;
      let addedCount = 0;
      let capHit = false;

      for (const file of accepted) {
        const validationError = validateImageFile(file);
        if (validationError) {
          if (!firstSkipReason) {
            firstSkipReason = `${file.name}: ${validationError}`;
          }
          continue;
        }

        try {
          await uploadSelfie(file);
          addedCount++;
        } catch (err) {
          if (err instanceof ApiError) {
            const code = err.errors[0]?.code;
            if (code === "SELFIE_LIMIT_REACHED") {
              capHit = true;
              break;
            }
            if (!firstSkipReason) {
              firstSkipReason = `${file.name}: ${err.errors[0]?.message ?? "Selfie rejected."}`;
            }
            continue;
          }
          if (!firstSkipReason) {
            firstSkipReason = `${file.name}: Could not upload.`;
          }
        }
      }

      // The cap rides the success toast rather than a red line beneath it. A
      // multi-pick past the limit used to fire both at once — green "Added 2
      // selfies" over red "Only 2 could be added" — two messages contradicting
      // each other about one action. The count also says out loud how many of
      // the picked files landed, which `files.slice(0, remaining)` above did
      // silently, in whatever order the OS handed them over.
      const capBlocked = capHit || files.length > remaining;

      if (addedCount > 0) {
        invalidateSelfieDependents();
        showToast({
          kind: "success",
          message: capBlocked
            ? `Added ${addedCount} of ${files.length} — ${SELFIE_MAX} selfies is the cap.`
            : addedCount === 1
              ? "Selfie added to your library."
              : `Added ${addedCount} selfies to your library.`,
        });
      } else if (capBlocked) {
        setError(
          `Library is full — ${SELFIE_MAX} selfies is the cap. Remove one to add another.`,
        );
      }

      // A rejected file is a different class from the cap: it names a file and
      // a reason, so it keeps the inline slot even when the toast fired. The
      // old `else if` chain dropped it whenever the pick was over cap too.
      if (firstSkipReason) setError(firstSkipReason);
    } catch {
      setError("Could not process one of the images. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove(id: string) {
    try {
      await deleteSelfie(id);
      invalidateSelfieDependents();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.errors[0]?.message ?? "Could not remove the selfie.")
          : "Could not remove the selfie.",
      );
      return;
    }
    showToast({ kind: "success", message: "Selfie removed." });
  }

  async function handleSetPrimary(id: string) {
    try {
      await setPrimarySelfie(id);
      invalidateSelfieDependents();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.errors[0]?.message ?? "Could not set primary.")
          : "Could not set primary. Try again.",
      );
      return;
    }
    showToast({ kind: "success", message: "Set as primary selfie." });
  }

  if (selfies.length === 0) {
    return <SelfieEmptyState onPick={handlePick} busy={busy} error={error} />;
  }

  return (
    <div>
      <div className="flex items-baseline justify-between gap-4 mb-5">
        <p className="font-sans text-base text-ink-soft max-w-md">
          {primary ? "Searches running across every event you join." : "Pick a primary selfie below."}{" "}
          <span className="text-slate">
            <span className="font-mono tnum text-ink">{selfies.length}</span>
            <span className="text-slate-soft"> / </span>
            <span className="font-mono tnum">{SELFIE_MAX}</span> selfies stored.
          </span>
        </p>
      </div>

      <ul className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 md:gap-4">
        {selfies.map((s) => (
          <li key={s.id}>
            <SelfieTile
              selfie={s}
              onRemove={() => handleRemove(s.id)}
              onSetPrimary={() => handleSetPrimary(s.id)}
            />
          </li>
        ))}
        <li>
          {remaining > 0 ? (
            <SelfieAddTile onPick={handlePick} busy={busy} />
          ) : (
            <SelfieCapTile />
          )}
        </li>
      </ul>

      {error && (
        <p role="alert" className="font-sans text-sm text-error mt-4">
          {error}
        </p>
      )}

      <p className="font-sans text-sm text-slate mt-6 max-w-md">
        Reused across every event you join — no re-uploading per race. Different
        angles improve match accuracy; clear, frontal shots work best.
      </p>
    </div>
  );
}

function SelfieEmptyState({
  onPick,
  busy,
  error,
}: {
  onPick: (e: ChangeEvent<HTMLInputElement>) => void;
  busy: boolean;
  error: string | null;
}) {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <SelfiePlusGlyph />
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink mt-5">
        Build your selfie library.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-md mx-auto">
        Upload up to <span className="font-mono tnum">{SELFIE_MAX}</span> clear,
        frontal selfies — we&apos;ll use them to find your photos at every event
        you join. Different angles improve match accuracy.
      </p>

      {error && (
        <p role="alert" className="font-sans text-sm text-error mt-4">
          {error}
        </p>
      )}

      <label
        className={
          "mt-6 inline-flex items-center gap-2 font-display text-base font-bold bg-fresh hover:bg-fresh-deep text-surface py-3 px-6 rounded-full transition-colors " +
          (busy ? "opacity-60 cursor-wait" : "cursor-pointer")
        }
      >
        {busy ? "Processing…" : "Upload selfies"}
        {!busy && <span aria-hidden="true">→</span>}
        <input
          type="file"
          accept={ACCEPTED_IMAGE_MIME.join(",")}
          onChange={onPick}
          multiple
          disabled={busy}
          className="sr-only"
        />
      </label>
    </div>
  );
}

function SelfieTile({
  selfie,
  onRemove,
  onSetPrimary,
}: {
  selfie: SelfieRef;
  onRemove: () => void;
  onSetPrimary: () => void;
}) {
  return (
    <div className="group relative aspect-square rounded-2xl overflow-hidden bg-ink">
      {/* eslint-disable-next-line @next/next/no-img-element -- selfie URLs are signed S3 outside Next image-domain config. */}
      <img
        src={selfie.dataUrl}
        alt=""
        className="size-full object-cover"
        draggable={false}
      />

      {selfie.isPrimary && (
        <span className="absolute top-2 left-2 inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-fresh text-surface font-mono uppercase tracking-[0.14em] text-[9px]">
          <span aria-hidden="true" className="size-1 rounded-full bg-bone" />
          Primary
        </span>
      )}

      {/* Never checked against ai-api, so it may not match once face search is
          live. Quiet ink scrim rather than an accent — `fresh` is spent on the
          Primary pill and the design system allows one per viewport. Absent
          once the selfie passes, so this disappears entirely when ai-api is on. */}
      {selfie.qualityTestStatus !== "passed" && (
        <span
          title="Uploaded before face matching was switched on — re-upload if search misses you."
          className="absolute top-2 right-2 px-2 py-1 rounded-full bg-ink/70 backdrop-blur text-bone font-mono uppercase tracking-[0.14em] text-[9px]"
        >
          Not checked
        </span>
      )}

      <div className="absolute inset-x-2 bottom-2 flex items-center justify-between gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity">
        {!selfie.isPrimary ? (
          <button
            type="button"
            onClick={onSetPrimary}
            className="font-mono uppercase tracking-[0.14em] text-[9px] bg-bone/90 backdrop-blur text-ink px-2 py-1 rounded-full hover:bg-bone transition-colors"
          >
            Make primary
          </button>
        ) : (
          <span aria-hidden="true" />
        )}
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove selfie"
          className="font-mono uppercase tracking-[0.14em] text-[9px] bg-bone/90 backdrop-blur text-ink px-2 py-1 rounded-full hover:bg-error hover:text-bone transition-colors"
        >
          Remove
        </button>
      </div>
    </div>
  );
}

function SelfieAddTile({
  onPick,
  busy,
}: {
  onPick: (e: ChangeEvent<HTMLInputElement>) => void;
  busy: boolean;
}) {
  return (
    <label
      className={
        "aspect-square rounded-2xl border-2 border-dashed border-line hover:border-fresh hover:bg-bone-deep/40 transition-colors flex flex-col items-center justify-center gap-2 text-slate hover:text-fresh " +
        (busy ? "opacity-60 cursor-wait" : "cursor-pointer")
      }
    >
      <span aria-hidden="true" className="text-2xl leading-none">
        +
      </span>
      <span className="font-mono uppercase tracking-[0.14em] text-[10px]">
        {busy ? "Processing" : "Add selfie"}
      </span>
      <input
        type="file"
        accept={ACCEPTED_IMAGE_MIME.join(",")}
        onChange={onPick}
        multiple
        disabled={busy}
        className="sr-only"
      />
    </label>
  );
}

// Holds the add-tile's slot at 5/5. Dropping the tile entirely left the grid
// with no affordance at all — the user saw five selfies and no explanation for
// why they couldn't add a sixth. Deliberately inert: no input, no hover state.
function SelfieCapTile() {
  return (
    <div className="aspect-square rounded-2xl border-2 border-dashed border-line bg-bone-deep/30 flex flex-col items-center justify-center gap-2 px-3 text-center">
      <Kicker tone="soft" tnum>
        {SELFIE_MAX} / {SELFIE_MAX}
      </Kicker>
      <span className="font-sans text-sm text-slate leading-snug">
        Remove one to add another.
      </span>
    </div>
  );
}

function SelfiePlusGlyph() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 64 64"
      fill="none"
      className="size-10 text-slate-soft mx-auto"
    >
      <circle cx="32" cy="24" r="10" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="M14 52c0-10 8-16 18-16s18 6 18 16"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle
        cx="50"
        cy="14"
        r="6"
        fill="var(--bone)"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path
        d="M50 11v6M47 14h6"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
    </svg>
  );
}
