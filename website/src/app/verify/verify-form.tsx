"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useMutation } from "@tanstack/react-query";
import { Kicker } from "@/components/ui/kicker";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ApiError, formatRetryWait } from "@/lib/api";
import { verifyPhoto, type PhotoVerifyResult } from "@/lib/api-verify";
import { cn } from "@/lib/utils";

// Public "whose photo is this?" page. One screen: drop an image, one CTA, one
// result card. The backend fingerprints the upload and answers with
// attribution only; nothing is stored.
const ACCEPT = ["image/jpeg", "image/png"];
const MAX_BYTES = 10 * 1024 * 1024;

export function VerifyForm() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const mutation = useMutation({ mutationFn: verifyPhoto });

  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file],
  );
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function pick(next: File | null) {
    mutation.reset();
    setLocalError(null);
    if (!next) {
      setFile(null);
      return;
    }
    if (!ACCEPT.includes(next.type)) {
      setLocalError("Use a JPEG or PNG — screenshots are usually PNG.");
      return;
    }
    if (next.size > MAX_BYTES) {
      setLocalError("That file is over 10 MB. A screenshot or preview is plenty.");
      return;
    }
    setFile(next);
  }

  const errorText =
    localError ?? (mutation.error ? describeError(mutation.error) : null);

  return (
    <section className="mx-auto w-full max-w-xl px-6 pt-12 pb-20 md:pt-20 md:pb-28">
      <Kicker as="p">Photo verification</Kicker>
      <h1 className="font-hero mt-3 text-[56px] leading-[0.92] md:text-[88px]">
        Whose photo
        <br />
        is this?
      </h1>
      <p className="mt-5 max-w-prose text-base leading-relaxed text-slate">
        Drop a screenshot or a saved QuickPitik preview. We compare a
        fingerprint of the picture against our library and tell you which
        photographer took it — even after a crop, a resize, or a re-save.
      </p>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          pick(e.dataTransfer.files?.[0] ?? null);
        }}
        className={cn(
          "mt-8 flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed bg-surface p-8 text-center transition-colors",
          "has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-fresh",
          dragging ? "border-ink bg-bone-deep" : "border-line-strong hover:border-ink",
        )}
      >
        <input
          type="file"
          accept={ACCEPT.join(",")}
          className="sr-only"
          onChange={(e) => pick(e.target.files?.[0] ?? null)}
        />
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt=""
            className="max-h-56 w-auto max-w-full rounded-xl object-contain"
          />
        ) : (
          <span
            aria-hidden="true"
            className="font-hero text-[44px] leading-none text-ink/15"
          >
            ?
          </span>
        )}
        <span className="font-display font-bold break-all">
          {file ? file.name : "Drop a screenshot here"}
        </span>
        <Kicker as="span" tone="soft">
          {file
            ? formatBytes(file.size)
            : "or tap to choose · JPEG / PNG · up to 10 MB"}
        </Kicker>
      </label>

      {errorText && (
        <p role="alert" className="mt-3 text-sm text-error">
          {errorText}
        </p>
      )}

      <button
        type="button"
        disabled={!file || mutation.isPending}
        onClick={() => file && mutation.mutate(file)}
        className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-5 w-full")}
      >
        {mutation.isPending ? "Checking…" : "Check this photo"}
      </button>

      <p className="mt-3 text-center text-sm text-slate-soft">
        We compare a fingerprint only — your upload is not stored.
      </p>

      {mutation.data && (
        <div className="mt-8" aria-live="polite">
          <ResultCard result={mutation.data} />
        </div>
      )}
    </section>
  );
}

function ResultCard({ result }: { result: PhotoVerifyResult }) {
  if (!result.matched) {
    return (
      <div className="animate-fade-in rounded-2xl border border-line bg-surface p-6">
        <Kicker as="p">No match</Kicker>
        <p className="mt-2 font-display text-xl font-bold">
          We couldn&rsquo;t match this image.
        </p>
        <p className="mt-2 text-sm leading-relaxed text-slate">
          Only QuickPitik previews are in the registry. Heavy crops, rotations,
          and filters break the fingerprint — try the original screenshot.
        </p>
      </div>
    );
  }

  const name = result.photographerName ?? "a QuickPitik photographer";
  const date = result.eventDate ? formatDate(result.eventDate) : null;

  return (
    <div className="animate-fade-in rounded-2xl border border-ink bg-surface p-6">
      <Kicker as="p">Match found · {result.confidence}</Kicker>
      <p className="mt-2 font-display text-xl font-bold leading-snug md:text-2xl">
        Yes, this is a QuickPitik photo.
      </p>
      <p className="mt-3 text-base leading-relaxed">
        Taken by <strong className="font-bold">{name}</strong>
        {result.photographerHandle && (
          <>
            {" "}
            <Link
              href={`/${result.photographerHandle}`}
              className="underline decoration-line-strong underline-offset-4 transition-colors hover:decoration-fresh"
            >
              @{result.photographerHandle}
            </Link>
          </>
        )}
        {result.eventName && <> at {result.eventName}</>}
        {date && (
          <>
            {" · "}
            <span className="tnum">{date}</span>
          </>
        )}
        .
      </p>
      {result.confidence === "weak" && (
        <p className="mt-3 text-sm leading-relaxed text-slate">
          Weak match: this copy has been cropped, edited, or re-encoded, so
          treat the attribution as likely rather than certain.
        </p>
      )}
    </div>
  );
}

function describeError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 429 && e.retryAfterSeconds != null) {
      return `Too many checks from this connection. Try again in ${formatRetryWait(e.retryAfterSeconds)}.`;
    }
    return e.message;
  }
  return "Couldn't reach QuickPitik — check your connection and try again.";
}

function formatBytes(bytes: number): string {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-PH", {
    year: "numeric",
    month: "long",
    day: "numeric",
  });
}
