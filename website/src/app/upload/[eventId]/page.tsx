"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { VerificationBanner } from "@/components/dashboard/verification-banner";
import { SiteHeader } from "@/components/layout/site-header";
import { ErrorBoundary } from "@/components/ui/error-boundary";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton } from "@/components/ui/skeleton";
import { useCanUpload } from "@/hooks/use-can-upload";
import { usePublicEvents } from "@/hooks/use-public-events";
import { useToast } from "@/hooks/use-toast";
import { usePhotographerVerificationSync } from "@/lib/photographer-verification-sync";
import {
  checkPhotosExist,
  sha256Hex,
  uploadPhotographerPhoto,
  type PhotoExistsResult,
} from "@/lib/api-photographer";
import { ROUTES } from "@/lib/constants";
import {
  UPLOAD_GRACE_DAYS,
  canUploadToEvent,
  useEventCatalog,
  uploadDaysRemaining,
} from "@/lib/event-catalog";
import { formatLongDate } from "@/lib/format";
import {
  ACCEPTED_IMAGE_MIME,
  HEIC_GUIDANCE,
  HEIC_REJECTION,
  isHeicFile,
  looksLikeHeic,
  MAX_UPLOAD_BYTES,
} from "@/lib/image-utils";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { cn } from "@/lib/utils";

// Auto-upload mode. Photos go straight to the gallery as they finish — there
// is no separate "publish" step. The page never redirects after a batch
// completes; instead, a fresh batch can be started in a new tab via the
// "Upload more" CTA. Same URL, fresh state — lets photographers fan out a
// 1500-photo coverage across three parallel tabs.

const BATCH_LIMIT = 500;

// Ceiling on simultaneous upload XHRs. Without it a 500-file batch fires 500
// requests at once — HTTP/2 multiplexes them onto one connection, so nothing
// queues at the transport layer and the tab holds 500 Files plus 500
// onprogress closures live. Mirrors the bounded worker pool the pre-flight
// hash step already uses below.
const MAX_CONCURRENT_UPLOADS = 4;

// Stable empty-array reference so useEventCatalog's memo doesn't churn while
// the public events fetch is in-flight.
const EMPTY_SEED: ReadonlyArray<ListEvent> = [];

const STATE_LABEL: Record<EventState, string> = {
  live: "LIVE",
  open: "OPEN",
  upcoming: "UPCOMING",
  past: "ARCHIVED",
};

interface UploadEntry {
  id: string;
  file: File;
  /** "checking" = local hash + backend pre-flight duplicate check in flight.
   *  "skipped"  = already live in THIS event (dedup Phase 2) — no upload sent.
   *  A "different event" duplicate becomes a non-retryable "error". */
  status: "checking" | "queued" | "uploading" | "done" | "error" | "skipped";
  /** 0..100 — mock progress, real backend will stream actual bytes. */
  progress: number;
  error?: string;
  /** True when the error happened during upload (network glitch — same file
   *  could succeed on a retry). False when the file was rejected by
   *  validation or a cross-event duplicate — same file means same error, so
   *  retry would mislead. */
  retryable?: boolean;
}

export default function FocusedUploadPage() {
  // Keep the suspended/verification state fresh while uploading — admin
  // suspend mid-session must show up here, not only on /dashboard/settings.
  usePhotographerVerificationSync();
  const params = useParams<{ eventId: string }>();
  const eventId = Array.isArray(params.eventId)
    ? params.eventId[0]
    : params.eventId;

  // Client-side fetch + admin-override merge. /upload/[eventId] is a client
  // page so it can't SSR; we read the public events list through React Query
  // and merge admin overrides on top. While the fetch is in flight, render
  // a skeleton — 404 only fires once we know the event truly isn't there.
  const liveEvents = usePublicEvents();
  const catalog = useEventCatalog(liveEvents ?? EMPTY_SEED);
  const event = useMemo(
    () => (eventId ? catalog.find((e) => e.id === eventId) : undefined),
    [catalog, eventId],
  );

  if (liveEvents === null) {
    return <FocusedUploadSkeleton />;
  }

  if (!event) {
    notFound();
  }

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />

      <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <BackChip />
        <Hero event={event} />
        <VerificationBanner />
        {/* Boundary scoped to the gate + dropzone so a thrown error in the
            upload pipeline (drag handler, batch validator, mock progress
            tick) keeps the back chip + hero usable. The header is outside
            the boundary so the user can always navigate away. */}
        <ErrorBoundary>
          <UploadGate event={event} />
        </ErrorBoundary>
      </div>
    </main>
  );
}

// Mirrors the loaded layout's shape (back chip → hero → dropzone) so the swap
// is reflow-free when the events fetch resolves. Same precedent as
// <FocusedShareSkeleton> on /dashboard/events/[id]. The back chip renders for
// real — it doesn't depend on the fetch, and a photographer who landed here by
// mistake shouldn't have to wait to leave.
function FocusedUploadSkeleton() {
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div
        className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24"
        aria-busy="true"
      >
        <BackChip />
        <section className="mb-12 md:mb-16">
          <Skeleton className="h-3 w-48" />
          <Skeleton className="h-10 md:h-14 w-3/4 mt-4" />
          <Skeleton className="h-4 w-1/3 mt-4" />
        </section>
        <Skeleton className="h-64 md:h-80 w-full rounded-2xl" />
      </div>
    </main>
  );
}

function BackChip() {
  return (
    <Link
      href={ROUTES.DASHBOARD_UPLOAD}
      className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[10px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone mb-8 md:mb-10"
    >
      <span aria-hidden="true">←</span>
      <span>Back to picker</span>
    </Link>
  );
}

function Hero({ event }: { event: ListEvent }) {
  return (
    <section className="mb-12 md:mb-16">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum flex items-center gap-2 flex-wrap">
        <span>{formatLongDate(event.date, true)}</span>
        <span className="text-slate-soft">·</span>
        <StateChip state={event.state} />
        <GraceBadge event={event} />
      </p>
      <h1 className="font-display text-4xl md:text-6xl font-medium tracking-tight text-ink mt-4 leading-[1.05]">
        {event.name}
      </h1>
      <p className="font-sans text-base md:text-lg text-ink-soft mt-4 max-w-md">
        {event.location}
      </p>
    </section>
  );
}

function StateChip({ state }: { state: EventState }) {
  if (state === "live") {
    return (
      <span className="inline-flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh breathe"
        />
        <span className="text-fresh">LIVE</span>
      </span>
    );
  }
  return <span>{STATE_LABEL[state]}</span>;
}

function UploadGate({ event }: { event: ListEvent }) {
  const gate = useCanUpload();

  if (event.state === "upcoming") {
    return (
      <div className="border border-line rounded-2xl px-6 py-12 bg-bone-deep/20 text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Not yet
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-3">
          Uploads open on race day.
        </p>
        <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto tnum">
          {formatLongDate(event.date, true)} · {event.location}
        </p>
        <p className="font-sans text-sm text-slate mt-6 max-w-md mx-auto">
          Pre-stage your kit now — the dropzone unlocks the moment the event
          flips to live.
        </p>
      </div>
    );
  }

  // Grace period closed — race day was 4+ days ago. Photographers cannot
  // push fresh photos to this event; the gallery itself stays open for sale
  // (and remains searchable on /events/[slug]).
  if (!canUploadToEvent(event.date)) {
    return (
      <div className="border border-line rounded-2xl px-6 py-12 bg-bone-deep/20 text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Upload window closed
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-3">
          Race-day grace ended.
        </p>
        <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto tnum">
          {formatLongDate(event.date, true)} · {event.location}
        </p>
        <p className="font-sans text-sm text-slate mt-6 max-w-md mx-auto">
          Photographers have a {UPLOAD_GRACE_DAYS}-day window starting on race
          day. Reach out to admin if you have late frames that need to land.
        </p>
        <Link
          href={ROUTES.DASHBOARD_UPLOAD}
          className="inline-flex mt-6 font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors"
        >
          Pick a different event
        </Link>
      </div>
    );
  }

  if (gate.kind === "suspended") {
    // The VerificationBanner above already carries the full suspended copy
    // + Contact support CTA; keep the dropzone block terse so the banner
    // owns the message.
    return (
      <div className="border border-line rounded-2xl px-6 py-12 bg-bone-deep/20 text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Uploads paused
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-3">
          Account suspended — see banner above.
        </p>
      </div>
    );
  }

  if (gate.kind !== "ok") {
    return (
      <div className="border border-line rounded-2xl px-6 py-12 bg-bone-deep/20 text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Uploads disabled
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-3">
          Finish verification to start uploading.
        </p>
      </div>
    );
  }

  return <UploadForm event={event} />;
}

function GraceBadge({ event }: { event: ListEvent }) {
  const remaining = uploadDaysRemaining(event.date);
  if (remaining === null) return null;
  const label =
    remaining === 0
      ? "Closes end of today"
      : remaining === 1
        ? "1 day left to upload"
        : `${remaining} days left to upload`;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="text-slate-soft">·</span>
      <span className="text-fresh tnum">{label}</span>
    </span>
  );
}

function UploadForm({ event }: { event: ListEvent }) {
  const { showToast } = useToast();
  const [entries, setEntries] = useState<UploadEntry[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);
  // Upload XHRs currently in flight — see MAX_CONCURRENT_UPLOADS.
  const activeRef = useRef(0);

  const accepted = useMemo(
    () => entries.filter((e) => e.status === "done"),
    [entries],
  );
  const errored = useMemo(
    () => entries.filter((e) => e.status === "error"),
    [entries],
  );
  const checking = useMemo(
    () => entries.filter((e) => e.status === "checking"),
    [entries],
  );
  const skipped = useMemo(
    () => entries.filter((e) => e.status === "skipped"),
    [entries],
  );
  const inFlight = useMemo(
    () =>
      entries.filter((e) => e.status === "queued" || e.status === "uploading"),
    [entries],
  );
  // Only files that will actually upload count toward the progress bar — a
  // "checking", "skipped" (already uploaded), or "error" file isn't sending
  // bytes, so including it would skew the percentage.
  const uploadBound = useMemo(
    () =>
      entries.filter(
        (e) =>
          e.status === "queued" ||
          e.status === "uploading" ||
          e.status === "done",
      ),
    [entries],
  );
  const overallPct = useMemo(() => {
    if (uploadBound.length === 0) return 0;
    const sum = uploadBound.reduce((acc, e) => acc + e.progress, 0);
    return Math.round(sum / uploadBound.length);
  }, [uploadBound]);

  const atLimit = entries.length >= BATCH_LIMIT;
  const newTabHref = `${ROUTES.UPLOAD}/${event.id}`;

  const ingestFiles = useCallback(
    (files: FileList | File[]) => {
      const incoming = Array.from(files);
      if (incoming.length === 0) return;

      const remaining = BATCH_LIMIT - entries.length;
      if (remaining <= 0) {
        showToast({
          kind: "error",
          message: `Batch is full at ${BATCH_LIMIT.toLocaleString()}. Open a new tab to keep uploading.`,
        });
        return;
      }

      const slice = incoming.slice(0, remaining);
      const dropped = incoming.length - slice.length;

      const next: UploadEntry[] = slice.map((file) => {
        const id = `${file.name}-${file.size}-${file.lastModified}`;
        const error = validate(file);
        return {
          id,
          file,
          // Valid files go to "checking" first — the pre-flight effect hashes
          // them and asks the backend which already exist before any upload.
          status: error ? "error" : "checking",
          progress: 0,
          error,
          retryable: false,
        };
      });

      setEntries((prev) => {
        const seen = new Set(prev.map((p) => p.id));
        const uniq = next.filter((n) => !seen.has(n.id));
        return [...prev, ...uniq];
      });

      // The row error has to stay short enough to survive `uppercase truncate`,
      // so the actual iPhone fix goes in a toast. Fired once per batch, not
      // once per file — a 200-photo camera-roll dump is all HEIC or none.
      if (slice.some(looksLikeHeic)) {
        showToast({ kind: "error", message: HEIC_GUIDANCE, duration: 6000 });
      }

      if (dropped > 0) {
        showToast({
          kind: "error",
          message: `${dropped.toLocaleString()} photo${dropped === 1 ? "" : "s"} skipped — over the ${BATCH_LIMIT.toLocaleString()}-per-batch limit.`,
          // Lengthen the dwell when the dropped count is larger than what a
          // glance can absorb in 4 s. 6 s is the longest dwell the toast
          // store currently honors without feeling sticky.
          duration: dropped > 5 ? 6000 : undefined,
        });
      }
    },
    [entries.length, showToast],
  );

  // Bumped on retry AND after the pre-flight check, so the upload effect
  // re-fires even when entries.length hasn't changed (retried entries flip
  // back to queued in place; checking entries flip forward to queued).
  const [retryNonce, setRetryNonce] = useState(0);

  // Pre-flight duplicate check (dedup Phase 2). Newly-ingested valid files land
  // as "checking"; here we hash them locally and ask the backend which already
  // exist for this photographer, so we never re-upload bytes that are already
  // stored. Each file is then routed:
  //   new             → "queued" (upload it)
  //   same_event      → "skipped" (already live in this event — no upload)
  //   different_event → non-retryable "error" (a photo can't live in two events)
  // The pre-flight is an optimization, not a gate: if it fails, every file
  // falls back to "queued" and the backend's unique index still dedupes.
  useEffect(() => {
    const pending = entries.filter((e) => e.status === "checking");
    if (pending.length === 0) return;
    let cancelled = false;

    void (async () => {
      // Hash with bounded concurrency so a 500-file batch doesn't read every
      // file into memory at once. The HEIC sniff rides along here rather than
      // in validate(): it has to read bytes, and this pass is already reading
      // every file. A HEIC is rejected outright, so skip hashing it.
      const files = pending.map((e) => e.file);
      const hashes = new Array<string>(files.length);
      const heic = new Array<boolean>(files.length);
      let cursor = 0;
      const worker = async () => {
        while (cursor < files.length) {
          const i = cursor++;
          heic[i] = await isHeicFile(files[i]);
          hashes[i] = heic[i] ? "" : await sha256Hex(files[i]);
        }
      };
      await Promise.all(
        Array.from({ length: Math.min(8, files.length) }, worker),
      );
      if (cancelled) return;

      let byHash = new Map<string, PhotoExistsResult>();
      const checkable = [...new Set(hashes.filter((h) => h.length > 0))];
      try {
        if (checkable.length > 0) {
          const results = await checkPhotosExist(event.id, checkable);
          byHash = new Map(results.map((r) => [r.hash, r]));
        }
      } catch {
        // Leave byHash empty → every file falls through to "queued" and the
        // server-side unique index remains the source of truth.
      }
      if (cancelled) return;

      const hashById = new Map(pending.map((e, i) => [e.id, hashes[i]]));
      const heicById = new Map(pending.map((e, i) => [e.id, heic[i]]));
      setEntries((prev) =>
        prev.map((p) => {
          if (p.status !== "checking") return p;
          const hash = hashById.get(p.id);
          if (hash === undefined) return p;
          if (heicById.get(p.id)) {
            return {
              ...p,
              status: "error",
              progress: 0,
              retryable: false,
              error: HEIC_REJECTION,
            };
          }
          const result = byHash.get(hash);
          if (result?.status === "same_event") {
            return { ...p, status: "skipped", progress: 0, error: undefined };
          }
          if (result?.status === "different_event") {
            return {
              ...p,
              status: "error",
              progress: 0,
              retryable: false,
              error: result.eventName
                ? `Already uploaded to "${result.eventName}".`
                : "Already uploaded to another event.",
            };
          }
          return { ...p, status: "queued" };
        }),
      );
      // Renamed HEICs only surface here — validate() saw image/jpeg and let
      // them through, so this is the photographer's first notice.
      if (heic.some(Boolean)) {
        showToast({ kind: "error", message: HEIC_GUIDANCE, duration: 6000 });
      }
      // checking → queued doesn't change entries.length, so nudge the upload
      // effect to pick the freshly-queued files up.
      setRetryNonce((n) => n + 1);
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries.length, event.id]);

  // Auto-upload: every queued file flows queued → uploading → done. There's
  // no separate publish step — done == in the gallery (web uploads go straight
  // to live; blur culling is desktop-only via BatchMyPhotos). One XHR per
  // file (Q-013) with onprogress driving setEntries.
  //
  // Only MAX_CONCURRENT_UPLOADS run at a time. `activeRef` counts the XHRs in
  // flight; each settled upload frees its slot and bumps retryNonce, which
  // re-runs this effect to start the next queued file. The queued → uploading
  // flip stays synchronous (before any await) so a re-fired effect can never
  // claim an entry that is already being sent.
  useEffect(() => {
    const slots = MAX_CONCURRENT_UPLOADS - activeRef.current;
    if (slots <= 0) return;
    const queued = entries
      .filter((e) => e.status === "queued")
      .slice(0, slots);
    if (queued.length === 0) return;

    queued.forEach((entry) => {
      activeRef.current += 1;
      setEntries((prev) =>
        prev.map((p) =>
          p.id === entry.id ? { ...p, status: "uploading" } : p,
        ),
      );
      uploadPhotographerPhoto(event.id, entry.file, (progress) => {
        setEntries((prev) =>
          prev.map((p) =>
            p.id === entry.id ? { ...p, progress: progress.percent } : p,
          ),
        );
      })
        .then(() => {
          setEntries((prev) =>
            prev.map((p) =>
              p.id === entry.id
                ? { ...p, status: "done", progress: 100 }
                : p,
            ),
          );
        })
        .catch((err: Error & { code?: string }) => {
          // Backend duplicate-rejection codes are terminal — the same bytes
          // will always be rejected, so don't offer a retry (mirrors the mobile
          // PhotoUploadWorker terminal-code guard). The pre-flight check catches
          // these first; this is the fallback for when it was skipped or failed.
          const terminal =
            err.code === "PHOTO_DUPLICATE_DIFFERENT_EVENT" ||
            err.code === "PHOTO_DUPLICATE_SAME_EVENT";
          setEntries((prev) =>
            prev.map((p) =>
              p.id === entry.id
                ? {
                    ...p,
                    status: "error",
                    error: err.message || "Upload failed.",
                    progress: 0,
                    retryable: !terminal,
                  }
                : p,
            ),
          );
        })
        .finally(() => {
          // Free the slot and nudge the effect so the next queued file starts.
          activeRef.current -= 1;
          setRetryNonce((n) => n + 1);
        });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries.length, retryNonce, event.id]);

  // Guard against losing a batch to F5 / browser-close / back-button. Photos
  // already marked done persist on the backend, but in-flight XHRs abort and
  // files still being hashed never get sent — both are gone with no record.
  // Browsers render their own confirmation copy; a custom string is ignored.
  const unsavedCount = inFlight.length + checking.length;
  useEffect(() => {
    if (unsavedCount === 0) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsavedCount]);

  function handleSelect(e: ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      ingestFiles(e.target.files);
    }
    e.target.value = "";
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = 0;
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      ingestFiles(e.dataTransfer.files);
    }
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
  }

  function handleDragEnter(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current += 1;
    setDragActive(true);
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    e.stopPropagation();
    dragCounter.current = Math.max(0, dragCounter.current - 1);
    if (dragCounter.current === 0) setDragActive(false);
  }

  const clearOne = useCallback((id: string) => {
    setEntries((prev) => prev.filter((p) => p.id !== id));
  }, []);

  function clearErrored() {
    setEntries((prev) => prev.filter((p) => p.status !== "error"));
  }

  // Reset a single retryable error back to queued so the upload effect picks
  // it up again. retryNonce++ guarantees the effect re-fires even though
  // entries.length hasn't changed.
  const retryOne = useCallback((id: string) => {
    setEntries((prev) =>
      prev.map((p) =>
        p.id === id && p.status === "error" && p.retryable
          ? { ...p, status: "queued", progress: 0, error: undefined }
          : p,
      ),
    );
    setRetryNonce((n) => n + 1);
  }, []);

  function retryAllFailed() {
    setEntries((prev) =>
      prev.map((p) =>
        p.status === "error" && p.retryable
          ? { ...p, status: "queued", progress: 0, error: undefined }
          : p,
      ),
    );
    setRetryNonce((n) => n + 1);
  }

  const retryableErroredCount = useMemo(
    () => entries.filter((e) => e.status === "error" && e.retryable).length,
    [entries],
  );

  return (
    <>
      <section className="mb-10">
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          className={cn(
            "rounded-2xl border-2 border-dashed transition-colors p-12 md:p-20 text-center",
            atLimit
              ? "border-error/50 bg-error/5"
              : dragActive
                ? "border-fresh bg-fresh/5"
                : "border-line bg-bone-deep/30 hover:bg-bone-deep/50",
          )}
        >
          <p
            className={cn(
              "font-display text-2xl md:text-3xl font-medium tracking-tight",
              atLimit ? "text-error" : "text-ink",
            )}
          >
            {atLimit ? "Batch full" : "Drop photos here"}
          </p>
          <p className="font-sans text-sm md:text-base text-slate mt-3 max-w-md mx-auto">
            {atLimit
              ? `${BATCH_LIMIT.toLocaleString()} is the maximum per batch. Open a new tab to keep uploading — this batch keeps its progress.`
              : `Photos go straight to your gallery as they finish uploading. Up to ${BATCH_LIMIT.toLocaleString()} per batch.`}
          </p>
          {atLimit ? (
            <a
              href={newTabHref}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-6 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
            >
              Upload more in new tab
              <span aria-hidden="true">↗</span>
            </a>
          ) : (
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-6 font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2"
            >
              Browse photos
              <span aria-hidden="true">→</span>
            </button>
          )}
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED_IMAGE_MIME.join(",")}
            multiple
            onChange={handleSelect}
            className="sr-only"
          />
        </div>

        <div className="mt-6 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono uppercase tracking-[0.25em] text-[10px] tnum">
          <span className={atLimit ? "text-error" : "text-slate-soft"}>
            Queued{" "}
            <span className={atLimit ? "text-error" : "text-ink"}>
              {entries.length.toLocaleString()}
            </span>{" "}
            / {BATCH_LIMIT.toLocaleString()}
          </span>
          {checking.length > 0 && (
            <span className="text-slate-soft">
              {checking.length.toLocaleString()} checking…
            </span>
          )}
          {skipped.length > 0 && (
            <span className="text-slate">
              {skipped.length.toLocaleString()} already uploaded
            </span>
          )}
          {errored.length > 0 && (
            <span className="text-error">
              {errored.length} skipped
              <button
                type="button"
                onClick={clearErrored}
                className="ml-2 underline decoration-1 underline-offset-2 hover:text-ink"
              >
                clear
              </button>
            </span>
          )}
        </div>
      </section>

      {entries.length > 0 && (
        <StagedSection
          entries={entries}
          done={accepted.length}
          inFlight={inFlight.length}
          validTotal={uploadBound.length}
          overallPct={overallPct}
          retryableErroredCount={retryableErroredCount}
          onClearAll={() => setEntries([])}
          onRemove={clearOne}
          onRetry={retryOne}
          onRetryAllFailed={retryAllFailed}
        />
      )}

      {entries.length > 0 && <UploadMoreFooter href={newTabHref} />}
    </>
  );
}

function StagedSection({
  entries,
  done,
  inFlight,
  validTotal,
  overallPct,
  retryableErroredCount,
  onClearAll,
  onRemove,
  onRetry,
  onRetryAllFailed,
}: {
  entries: UploadEntry[];
  done: number;
  inFlight: number;
  validTotal: number;
  overallPct: number;
  retryableErroredCount: number;
  onClearAll: () => void;
  onRemove: (id: string) => void;
  onRetry: (id: string) => void;
  onRetryAllFailed: () => void;
}) {
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.STAGED_INITIAL);
  const visibleSlice = entries.slice(0, loadedCount);

  return (
    <section className="mb-10">
      <div className="flex items-baseline justify-between border-b border-line pb-4 mb-6 gap-4 flex-wrap">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum">
          <span className="text-ink">{entries.length.toLocaleString()}</span>{" "}
          staged
        </p>
        <div className="flex items-baseline gap-5">
          {retryableErroredCount > 0 && (
            <button
              type="button"
              onClick={onRetryAllFailed}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
            >
              Retry{" "}
              <span className="tnum">
                {retryableErroredCount.toLocaleString()}
              </span>{" "}
              failed
            </button>
          )}
          <button
            type="button"
            onClick={onClearAll}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
          >
            Clear all
          </button>
        </div>
      </div>

      {/* A batch can be entirely "already uploaded" (re-dragged folder) — then
          nothing is upload-bound and a 0 / 0 bar would read as broken. */}
      {validTotal > 0 && (
        <OverallProgress
          done={done}
          inFlight={inFlight}
          validTotal={validTotal}
          pct={overallPct}
        />
      )}

      <ul className="divide-y divide-line">
        {visibleSlice.map((entry) => (
          <li key={entry.id}>
            <QueueRow entry={entry} onRemove={onRemove} onRetry={onRetry} />
          </li>
        ))}
      </ul>
      <LoadMoreButton
        shown={visibleSlice.length}
        total={entries.length}
        increment={PAGE_SIZE.STAGED_INCREMENT}
        onLoadMore={() =>
          setLoadedCount((n) => n + PAGE_SIZE.STAGED_INCREMENT)
        }
      />
    </section>
  );
}

function OverallProgress({
  done,
  inFlight,
  validTotal,
  pct,
}: {
  done: number;
  inFlight: number;
  validTotal: number;
  pct: number;
}) {
  const isComplete = pct >= 100 && validTotal > 0;
  return (
    <div className="mb-8">
      <div className="flex items-baseline justify-between gap-4 mb-3">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] tnum">
          {isComplete ? (
            <span className="text-fresh">
              All uploaded · {validTotal.toLocaleString()} live
            </span>
          ) : (
            <span className="text-slate">
              <span className="text-ink">{done.toLocaleString()}</span>
              <span className="text-slate-soft"> / </span>
              <span>{validTotal.toLocaleString()}</span>
              <span> live</span>
              {inFlight > 0 && (
                <>
                  <span className="text-slate-soft"> · </span>
                  <span>{inFlight.toLocaleString()} uploading</span>
                </>
              )}
            </span>
          )}
        </p>
        <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-ink tnum">
          {pct}%
        </p>
      </div>
      <div className="h-1 bg-bone-deep rounded-full overflow-hidden">
        <div
          className={cn(
            "h-full transition-[width] duration-300 ease-out",
            isComplete ? "bg-fresh" : "bg-ink-soft",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function UploadMoreFooter({ href }: { href: string }) {
  return (
    <section className="mt-12 md:mt-16 border-t border-line pt-8 flex flex-col md:flex-row md:items-center md:justify-between gap-6">
      <div className="max-w-md">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
          Need to upload more?
        </p>
        <p className="font-sans text-base text-ink-soft mt-2">
          Open a fresh batch in a new tab — this tab keeps its history so you
          can track both at once.
        </p>
      </div>
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center justify-center gap-2 shrink-0"
      >
        Upload more in new tab
        <span aria-hidden="true">↗</span>
      </a>
    </section>
  );
}

// Memoized so off-batch rows don't re-render on every progress tick. A 500-photo
// batch ticks setEntries every ~90 ms; without memo, every row re-renders each
// tick (12k renders/sec). With memo + identity-stable onRemove/onRetry, only
// the row whose entry reference changed re-renders.
const QueueRow = memo(function QueueRow({
  entry,
  onRemove,
  onRetry,
}: {
  entry: UploadEntry;
  onRemove: (id: string) => void;
  onRetry: (id: string) => void;
}) {
  const status = entry.status;
  const canRetry = status === "error" && entry.retryable === true;

  // Local object URL for the row thumbnail. Bound to the effect lifecycle so
  // React 18 StrictMode's double-mount in dev (mount → cleanup → mount) creates
  // a fresh blob on the second mount instead of reusing a memoized handle the
  // cleanup already revoked — which would surface as net::ERR_FILE_NOT_FOUND
  // on the first paint. Cheap memory cost: a browser-side handle to the
  // in-memory File bytes, no network round-trip and no copy. Backend-rendered
  // thumbs are delivered after upload completes but aren't surfaced on this
  // page — photographers want immediate confirmation of "right photo, right slot."
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  useEffect(() => {
    const url = URL.createObjectURL(entry.file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [entry.file]);

  return (
    <div className="py-4 md:py-5 flex items-center gap-4">
      <div
        className={cn(
          "size-10 md:size-12 rounded-md border border-line shrink-0 overflow-hidden relative",
          status === "error" ? "bg-error/10" : "bg-bone-deep/40",
        )}
      >
        {previewUrl && (
          /* eslint-disable-next-line @next/next/no-img-element */
          <img
            src={previewUrl}
            alt=""
            className={cn(
              "absolute inset-0 w-full h-full object-cover",
              status === "error" ? "opacity-40" : "opacity-100",
            )}
            draggable={false}
          />
        )}
        {(status === "queued" ||
          status === "uploading" ||
          status === "checking") && (
          <span
            aria-hidden="true"
            className="absolute inset-0 flex items-center justify-center bg-ink/30 font-mono text-[10px] tracking-[0.15em] uppercase text-bone tnum"
          >
            …
          </span>
        )}
        {status === "done" && (
          <span
            aria-hidden="true"
            className="absolute bottom-0.5 right-0.5 size-3.5 rounded-full bg-fresh text-bone flex items-center justify-center font-mono text-[8px] leading-none"
          >
            ✓
          </span>
        )}
        {/* Muted (slate, not fresh) check — already in the gallery, not a
            this-batch upload. */}
        {status === "skipped" && (
          <span
            aria-hidden="true"
            className="absolute bottom-0.5 right-0.5 size-3.5 rounded-full bg-slate text-bone flex items-center justify-center font-mono text-[8px] leading-none"
          >
            ✓
          </span>
        )}
        {status === "error" && (
          <span
            aria-hidden="true"
            className="absolute inset-0 flex items-center justify-center font-mono text-[12px] font-bold text-error"
          >
            !
          </span>
        )}
      </div>

      <div className="flex-1 min-w-0">
        <p className="font-sans text-sm md:text-base text-ink truncate">
          {entry.file.name}
        </p>
        <div className="mt-1 flex items-center gap-3">
          <p className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum uppercase shrink-0">
            {formatBytes(entry.file.size)}
          </p>
          {status === "uploading" || status === "queued" ? (
            <div className="flex-1 h-0.5 bg-bone-deep rounded-full overflow-hidden">
              <div
                className="h-full bg-ink-soft transition-[width] duration-150 ease-out"
                style={{ width: `${entry.progress}%` }}
              />
            </div>
          ) : status === "checking" ? (
            <p className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum uppercase">
              Checking…
            </p>
          ) : status === "done" ? (
            <p className="font-mono text-[10px] tracking-[0.15em] text-fresh tnum uppercase">
              Live
            </p>
          ) : status === "skipped" ? (
            <p className="font-mono text-[10px] tracking-[0.15em] text-slate tnum uppercase">
              Already uploaded
            </p>
          ) : (
            <p className="font-mono text-[10px] tracking-[0.15em] text-error tnum uppercase truncate">
              {entry.error ?? "Failed"}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center gap-1 shrink-0">
        {canRetry && (
          <button
            type="button"
            onClick={() => onRetry(entry.id)}
            aria-label={`Try uploading ${entry.file.name} again`}
            className="font-mono text-[10px] tracking-[0.15em] uppercase text-slate hover:text-ink transition-colors px-2 py-1"
          >
            Try again
          </button>
        )}
        <button
          type="button"
          onClick={() => onRemove(entry.id)}
          aria-label={`Remove ${entry.file.name}`}
          className="font-mono text-[10px] tracking-[0.15em] uppercase text-slate hover:text-error transition-colors px-2 py-1"
        >
          Remove
        </button>
      </div>
    </div>
  );
});

function validate(file: File): string | undefined {
  // Named HEIC is caught here; a HEIC renamed to .jpg reports image/jpeg and
  // slips past — the pre-flight pass sniffs its header and rejects it there.
  // Both paths share HEIC_GUIDANCE so the copy doesn't diverge.
  if (looksLikeHeic(file)) {
    return HEIC_REJECTION;
  }
  if (!ACCEPTED_IMAGE_MIME.includes(file.type)) {
    return "JPEG, PNG, or WebP only.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "Larger than 8 MB.";
  }
  return undefined;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
