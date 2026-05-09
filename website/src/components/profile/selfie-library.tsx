"use client";

import { useState, type ChangeEvent } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  SELFIE_MAX,
  useUserMediaStore,
  type SelfieRef,
} from "@/store/user-media-store";
import { useSelfiesList } from "@/hooks/use-selfies";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import {
  uploadSelfie,
  deleteSelfie,
  setPrimarySelfie,
} from "@/lib/api-selfies";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  ACCEPTED_IMAGE_MIME,
  fitToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";

const SELFIE_LONG_EDGE_PX = 1024;

export function SelfieLibrary() {
  const { selfies } = useSelfiesList();
  const addSelfieMock = useUserMediaStore((s) => s.addSelfie);
  const removeSelfieMock = useUserMediaStore((s) => s.removeSelfie);
  const setPrimaryMock = useUserMediaStore((s) => s.setPrimary);
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const remaining = SELFIE_MAX - selfies.length;
  const primary = selfies.find((s) => s.isPrimary);

  function invalidateSelfieDependents() {
    queryClient.invalidateQueries({ queryKey: ["me", "selfies"] });
    // Selfie-search results depend on the user's primary embedding; refresh
    // any cached event photo queries so the cockpit picks up the new primary.
    queryClient.invalidateQueries({
      predicate: (q) =>
        Array.isArray(q.queryKey) &&
        q.queryKey[0] === "events" &&
        q.queryKey[2] === "photos",
    });
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

        if (BACKEND_LIVE) {
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
        } else {
          const dataUrl = await fitToDataUrl(file, SELFIE_LONG_EDGE_PX);
          addSelfieMock({
            id: crypto.randomUUID(),
            dataUrl,
            uploadedAt: new Date().toISOString(),
            isPrimary: false,
            qualityScore: 0.92,
          });
          addedCount++;
        }
      }

      if (BACKEND_LIVE && addedCount > 0) {
        invalidateSelfieDependents();
      }

      if (addedCount > 0) {
        showToast({
          kind: "success",
          message:
            addedCount === 1
              ? "Selfie added to your library."
              : `Added ${addedCount} selfies to your library.`,
        });
      }

      if (capHit) {
        setError(`Reached the ${SELFIE_MAX}-selfie cap.`);
      } else if (files.length > remaining) {
        setError(
          `Only ${remaining} selfie${remaining === 1 ? "" : "s"} could be added — ${SELFIE_MAX} is the cap.`,
        );
      } else if (firstSkipReason) {
        setError(firstSkipReason);
      }
    } catch {
      setError("Could not process one of the images. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove(id: string) {
    if (BACKEND_LIVE) {
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
    } else {
      removeSelfieMock(id);
    }
    showToast({ kind: "success", message: "Selfie removed." });
  }

  async function handleSetPrimary(id: string) {
    if (BACKEND_LIVE) {
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
    } else {
      setPrimaryMock(id);
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
        {remaining > 0 && (
          <li>
            <SelfieAddTile onPick={handlePick} busy={busy} />
          </li>
        )}
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
          "mt-6 inline-flex items-center gap-2 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors " +
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
      {/* eslint-disable-next-line @next/next/no-img-element -- selfie URLs (signed S3 in live, base64 in mock) outside Next image-domain config. */}
      <img
        src={selfie.dataUrl}
        alt=""
        className="size-full object-cover"
        draggable={false}
      />

      {selfie.isPrimary && (
        <span className="absolute top-2 left-2 inline-flex items-center gap-1.5 px-2 py-1 rounded-full bg-fresh text-bone font-mono uppercase tracking-[0.2em] text-[9px]">
          <span aria-hidden="true" className="size-1 rounded-full bg-bone" />
          Primary
        </span>
      )}

      <div className="absolute inset-x-2 bottom-2 flex items-center justify-between gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity">
        {!selfie.isPrimary ? (
          <button
            type="button"
            onClick={onSetPrimary}
            className="font-mono uppercase tracking-[0.2em] text-[9px] bg-bone/90 backdrop-blur text-ink px-2 py-1 rounded-full hover:bg-bone transition-colors"
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
          className="font-mono uppercase tracking-[0.2em] text-[9px] bg-bone/90 backdrop-blur text-ink px-2 py-1 rounded-full hover:bg-error hover:text-bone transition-colors"
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
      <span className="font-mono uppercase tracking-[0.25em] text-[10px]">
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
