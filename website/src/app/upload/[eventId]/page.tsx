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
import { useCanUpload } from "@/hooks/use-can-upload";
import { usePublicEvents } from "@/hooks/use-public-events";
import { useToast } from "@/hooks/use-toast";
import { usePhotographerVerificationSync } from "@/lib/photographer-verification-sync";
import { uploadPhotographerPhoto } from "@/lib/api-photographer";
import { ROUTES } from "@/lib/constants";
import {
  UPLOAD_GRACE_DAYS,
  canUploadToEvent,
  useEventCatalog,
  uploadDaysRemaining,
} from "@/lib/event-catalog";
import { formatLongDate } from "@/lib/format";
import { ACCEPTED_IMAGE_MIME, MAX_UPLOAD_BYTES } from "@/lib/image-utils";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { cn } from "@/lib/utils";

// Auto-upload mode. Photos go straight to the gallery as they finish — there
// is no separate "publish" step. The page never redirects after a batch
// completes; instead, a fresh batch can be started in a new tab via the
// "Upload more" CTA. Same URL, fresh state — lets photographers fan out a
// 1500-photo coverage across three parallel tabs.

const BATCH_LIMIT = 500;

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
  status: "queued" | "uploading" | "done" | "error";
  /** 0..100 — mock progress, real backend will stream actual bytes. */
  progress: number;
  error?: string;
  /** True when the error happened during upload (network glitch — same file
   *  could succeed on a retry). False when the file was rejected by
   *  validation — same file means same error, so retry would mislead. */
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
    return (
      <main className="bg-bone text-ink min-h-screen flex flex-col">
        <SiteHeader />
        <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
            Loading event…
          </p>
        </div>
      </main>
    );
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

  const accepted = useMemo(
    () => entries.filter((e) => e.status === "done"),
    [entries],
  );
  const errored = useMemo(
    () => entries.filter((e) => e.status === "error"),
    [entries],
  );
  const inFlight = useMemo(
    () =>
      entries.filter((e) => e.status === "queued" || e.status === "uploading"),
    [entries],
  );
  const validEntries = useMemo(
    () => entries.filter((e) => e.status !== "error"),
    [entries],
  );
  const overallPct = useMemo(() => {
    if (validEntries.length === 0) return 0;
    const sum = validEntries.reduce((acc, e) => acc + e.progress, 0);
    return Math.round(sum / validEntries.length);
  }, [validEntries]);

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
          status: error ? "error" : "queued",
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

  // Bumped on retry so the upload effect re-fires even when entries.length
  // hasn't changed (retried entries flip back to queued in place).
  const [retryNonce, setRetryNonce] = useState(0);

  // Auto-upload: every queued file flows queued → uploading → done. There's
  // no separate publish step — done == in the gallery (web uploads go straight
  // to live; blur culling is desktop-only via BatchMyPhotos). One XHR per
  // file (Q-013) with onprogress driving setEntries.
  useEffect(() => {
    const queued = entries.filter((e) => e.status === "queued");
    if (queued.length === 0) return;

    queued.forEach((entry) => {
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
        .catch((err: Error) => {
          setEntries((prev) =>
            prev.map((p) =>
              p.id === entry.id
                ? {
                    ...p,
                    status: "error",
                    error: err.message || "Upload failed.",
                    progress: 0,
                    retryable: true,
                  }
                : p,
            ),
          );
        });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [entries.length, retryNonce, event.id]);

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
          validTotal={validEntries.length}
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

      <OverallProgress
        done={done}
        inFlight={inFlight}
        validTotal={validTotal}
        pct={overallPct}
      />

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

  return (
    <div className="py-4 md:py-5 flex items-center gap-4">
      <div
        className={cn(
          "size-10 md:size-12 rounded-md border border-line shrink-0 flex items-center justify-center",
          status === "error" ? "bg-error/10" : "bg-bone-deep/40",
        )}
        aria-hidden="true"
      >
        <span
          className={cn(
            "font-mono text-[9px] tracking-[0.15em] uppercase tnum",
            status === "error" ? "text-error" : "text-slate",
          )}
        >
          {status === "done" ? "✓" : status === "error" ? "!" : "…"}
        </span>
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
          ) : status === "done" ? (
            <p className="font-mono text-[10px] tracking-[0.15em] text-fresh tnum uppercase">
              Live
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
