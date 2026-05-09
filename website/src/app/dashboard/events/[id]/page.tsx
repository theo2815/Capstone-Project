"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useCallback, useMemo, useState } from "react";
import type { EventState, ListEvent } from "@/app/events/events-browser";
import { SiteHeader } from "@/components/layout/site-header";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Skeleton, TileSkeleton } from "@/components/ui/skeleton";
import {
  usePhotographerEventDetail,
  usePhotographerEventPhotos,
} from "@/hooks/use-photographer-data";
import { useToast } from "@/hooks/use-toast";
import { fetchPhotographerPhotoDownload } from "@/lib/api-photographer";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { ROUTES } from "@/lib/constants";
import { getEventById } from "@/lib/event-catalog";
import { formatLongDate } from "@/lib/format";
import { useMockLatency } from "@/lib/mock-latency";
import {
  PHOTOGRAPHER_EVENTS,
  generatePhotographerLibrary,
  type PhotographerEventSummary,
  type PhotographerLibraryPhoto,
} from "@/lib/photographer-mock";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { cn } from "@/lib/utils";

// Focused share page. Direction A composition: SiteHeader (no dashboard rail
// — escaped via the conditional in app/dashboard/layout.tsx) → back chip →
// event hero → Share Hero Band (public URL + copy + social share chips) →
// stats row → photo grid. Photo grid mirrors /events/[slug]?browse=1's
// mosaic (row-span-1 wide / row-span-2 default, fixed grid-auto-rows) so
// landscape uploads sit naturally without letterboxing. Click any tile to
// open <PhotoPreviewCard mode="owned"> — photographer-side, no cart/buy
// footer, just a Download CTA.

const TONE_COLORS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
];

const STATE_LABEL: Record<EventState, string> = {
  live: "LIVE",
  open: "OPEN",
  upcoming: "UPCOMING",
  past: "ARCHIVED",
};

export default function FocusedSharePage() {
  const params = useParams<{ id: string }>();
  const id = Array.isArray(params.id) ? params.id[0] : params.id;

  // Live mode prefers GET /me/photographer/events/{id} (Q-014); mock-mode
  // falls back to the existing PHOTOGRAPHER_EVENTS seed lookup.
  const liveDetail = usePhotographerEventDetail(id ?? null);
  const livePhotos = usePhotographerEventPhotos(id ?? null);

  // Memoize the synchronous lookup so useMockLatency sees a stable reference
  // per id — without it, every parent re-render hands the hook a fresh object
  // and the effect re-fires unnecessarily.
  const resolved = useMemo(
    () => ({
      event: id ? getEventById(id) : undefined,
      photographer:
        liveDetail ??
        (id ? PHOTOGRAPHER_EVENTS.find((p) => p.id === id) : undefined),
    }),
    [id, liveDetail],
  );
  const { data, isLoading } = useMockLatency(resolved);

  // Show skeleton while the (mock-latency-gated) lookup is in flight. Once
  // resolved with no event / no photographer / no uploads → 404.
  if (isLoading || data === null) {
    return <FocusedShareSkeleton />;
  }

  const { event, photographer } = data;

  // Only events the photographer has actually covered render this page.
  // Anything else (catalog event without uploads, or no event at all) → 404.
  if (!event || !photographer || photographer.photoCount === 0) {
    notFound();
  }

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <BackChip />
        <Hero event={event} />
        <ShareHeroBand event={event} />
        <Stats photographer={photographer} />
        <PhotoGrid
          event={event}
          photographer={photographer}
          livePhotos={livePhotos}
        />
      </div>
    </main>
  );
}

// Pulse-animated placeholder mirroring the share page's actual layout shape
// (back chip → hero → share band → stats → mosaic) so swap-in is reflow-free
// when data resolves. Mounts behind useMockLatency — invisible at the dev
// default of MOCK_LATENCY_MS = 0.
function FocusedShareSkeleton() {
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <Skeleton className="h-3 w-32 mb-8 md:mb-10" />
        <section className="mb-10 md:mb-12">
          <Skeleton className="h-3 w-44 mb-4" />
          <Skeleton className="h-12 md:h-16 w-3/4 mb-4" />
          <Skeleton className="h-4 w-1/2" />
        </section>
        <Skeleton className="h-14 w-full rounded-2xl mb-8 md:mb-12" />
        <div className="flex flex-wrap gap-x-8 gap-y-4 mb-10 md:mb-12">
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-8 w-24" />
          <Skeleton className="h-8 w-24" />
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 md:gap-4">
          {Array.from({ length: 8 }).map((_, i) => (
            <TileSkeleton
              key={i}
              aspectRatio={i % 3 === 0 ? "aspect-[3/4]" : "aspect-[3/2]"}
            />
          ))}
        </div>
      </div>
    </main>
  );
}

function BackChip() {
  return (
    <Link
      href={ROUTES.DASHBOARD_EVENTS}
      className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[10px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone mb-8 md:mb-10"
    >
      <span aria-hidden="true">←</span>
      <span>Back to events</span>
    </Link>
  );
}

function Hero({ event }: { event: ListEvent }) {
  return (
    <section className="mb-10 md:mb-12">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum flex items-center gap-2 flex-wrap">
        <span>{formatLongDate(event.date, true)}</span>
        <span className="text-slate-soft">·</span>
        <StateChip state={event.state} />
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

function ShareHeroBand({ event }: { event: ListEvent }) {
  const handle = usePhotographerSettingsStore((s) => s.handle) || "your-handle";
  const { showToast } = useToast();
  const [copied, setCopied] = useState(false);

  const path = `/${handle}/events/${event.slug}`;
  const fullUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}${path}`
      : `https://quickpitik.com${path}`;
  const displayUrl = `quickpitik.com${path}`;

  function handleCopy() {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard.writeText(fullUrl);
    setCopied(true);
    showToast({ kind: "success", message: "Link copied." });
    setTimeout(() => setCopied(false), 2500);
  }

  function shareTo(provider: "facebook" | "x" | "threads") {
    const encoded = encodeURIComponent(fullUrl);
    const tweetText = encodeURIComponent(`Photos from ${event.name}`);
    const threadsText = encodeURIComponent(
      `Photos from ${event.name} — ${displayUrl}`,
    );
    let url = "";
    switch (provider) {
      case "facebook":
        url = `https://www.facebook.com/sharer/sharer.php?u=${encoded}`;
        break;
      case "x":
        url = `https://twitter.com/intent/tweet?url=${encoded}&text=${tweetText}`;
        break;
      case "threads":
        url = `https://www.threads.net/intent/post?text=${threadsText}`;
        break;
    }
    window.open(url, "_blank", "noopener,noreferrer,width=600,height=480");
  }

  function handleInstagram() {
    if (typeof navigator === "undefined" || !navigator.clipboard) return;
    navigator.clipboard.writeText(fullUrl);
    showToast({
      kind: "info",
      message: "Link copied. Paste into your IG bio or story.",
    });
  }

  function handleNativeShare() {
    if (typeof navigator !== "undefined" && navigator.share) {
      navigator
        .share({
          title: event.name,
          text: `Photos from ${event.name}`,
          url: fullUrl,
        })
        .catch(() => {
          // user cancelled, ignore
        });
    } else {
      handleCopy();
    }
  }

  return (
    <section className="mb-10 md:mb-12 border border-line rounded-2xl bg-bone-deep/30 p-6 md:p-10">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        Public gallery
      </p>

      <div className="mt-4 flex flex-col md:flex-row md:items-center gap-3 md:gap-4">
        <code className="flex-1 min-w-0 font-mono text-sm md:text-base text-ink truncate bg-bone-deep/60 border border-line rounded-full px-4 py-2.5">
          {displayUrl}
        </code>
        <button
          type="button"
          onClick={handleCopy}
          aria-label="Copy public gallery link"
          className="font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors inline-flex items-center justify-center gap-2 shrink-0"
        >
          {copied ? "Copied!" : "Copy link"}
        </button>
      </div>

      <div className="mt-6 md:mt-8 flex flex-col md:flex-row md:items-center gap-3 md:gap-6">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft shrink-0">
          Share to your followers
        </p>
        <div className="flex items-center gap-2 md:gap-3 flex-wrap">
          <ShareChip
            onClick={() => shareTo("facebook")}
            dotColor="#1877F2"
            label="Facebook"
          />
          <ShareChip
            onClick={handleInstagram}
            dotColor="#E4405F"
            label="Instagram"
          />
          <ShareChip
            onClick={() => shareTo("x")}
            dotColor="#000000"
            label="X"
          />
          <ShareChip
            onClick={() => shareTo("threads")}
            dotColor="#444444"
            label="Threads"
          />
          <ShareChip
            onClick={handleNativeShare}
            dotColor="var(--slate-soft)"
            label="More"
          />
        </div>
      </div>
    </section>
  );
}

function ShareChip({
  onClick,
  dotColor,
  label,
}: {
  onClick: () => void;
  dotColor: string;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={`Share to ${label}`}
      className="inline-flex items-center gap-2 font-sans text-sm font-medium border border-line bg-bone hover:bg-bone-deep hover:border-ink py-2 px-4 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <span
        aria-hidden="true"
        className="size-2 rounded-full shrink-0"
        style={{ backgroundColor: dotColor }}
      />
      <span>{label}</span>
    </button>
  );
}

function Stats({ photographer }: { photographer: PhotographerEventSummary }) {
  return (
    <section className="mb-10 md:mb-12 flex flex-wrap items-baseline gap-x-8 gap-y-3 font-mono uppercase tracking-[0.3em] text-[11px] text-slate tnum">
      <span>
        <span className="text-ink">
          {photographer.photoCount.toLocaleString()}
        </span>{" "}
        photos
      </span>
      <span>
        <span className="text-ink">
          {photographer.salesCount.toLocaleString()}
        </span>{" "}
        sold
      </span>
      <span>
        <span className="text-ink">
          ₱{photographer.revenueKept.toLocaleString()}
        </span>{" "}
        kept
      </span>
    </section>
  );
}

function PhotoGrid({
  event,
  photographer,
  livePhotos,
}: {
  event: ListEvent;
  photographer: PhotographerEventSummary;
  livePhotos: PhotographerLibraryPhoto[] | null;
}) {
  // Live mode prefers GET /me/photographer/events/{id}/photos (Q-014).
  // Mock-mode falls back to the deterministic library generator.
  const photos = useMemo(
    () => livePhotos ?? generatePhotographerLibrary(photographer),
    [livePhotos, photographer],
  );
  const total = photos.length;

  const { showToast } = useToast();
  const [openIndex, setOpenIndex] = useState<number | null>(null);
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.PHOTO_INITIAL);

  const visibleSlice = useMemo(
    () => photos.slice(0, loadedCount),
    [photos, loadedCount],
  );

  const handleOpen = useCallback((i: number) => setOpenIndex(i), []);
  const handleClose = useCallback(() => setOpenIndex(null), []);
  const handlePrev = useCallback(() => {
    setOpenIndex((i) => (i === null || i === 0 ? i : i - 1));
  }, []);
  const handleNext = useCallback(() => {
    setOpenIndex((i) =>
      i === null || i === visibleSlice.length - 1 ? i : i + 1,
    );
  }, [visibleSlice.length]);
  const handleDownload = useCallback(async () => {
    const photoId =
      openIndex !== null ? visibleSlice[openIndex]?.id : undefined;
    if (!photoId) return;
    // Live mode: signed-URL fetch + programmatic <a download> click (Q-015,
    // 5-min TTL). Mock mode shows the success toast without a real fetch.
    if (BACKEND_LIVE) {
      try {
        const result = await fetchPhotographerPhotoDownload(photoId);
        if (result?.url) {
          const a = document.createElement("a");
          a.href = result.url;
          a.download = "";
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          showToast({ kind: "success", message: "Download started." });
          return;
        }
      } catch (err) {
        console.error("[photographer/download] failed", err);
        showToast({ kind: "error", message: "Download failed. Try again." });
        return;
      }
    }
    showToast({
      kind: "success",
      message: "Download started. (mock)",
    });
  }, [openIndex, visibleSlice, showToast]);

  const openPhoto = openIndex !== null ? visibleSlice[openIndex] : null;
  const previewItem: PhotoPreviewItem | null = openPhoto
    ? {
        id: openPhoto.id,
        bib: openPhoto.bib,
        time: "",
        tone: openPhoto.tone,
        price: 0,
        span: openPhoto.span,
        imageUrl: null,
      }
    : null;

  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-line pb-4 mb-6 gap-4">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum">
          Uploaded photos
        </p>
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft tnum">
          <span className="text-ink">{total.toLocaleString()}</span> total
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
        {visibleSlice.map((photo, i) => (
          <PhotoTile
            key={photo.id}
            photo={photo}
            index={i}
            onOpen={() => handleOpen(i)}
          />
        ))}
      </div>

      <LoadMoreButton
        shown={visibleSlice.length}
        total={total}
        increment={PAGE_SIZE.PHOTO_INCREMENT}
        onLoadMore={() => setLoadedCount((n) => n + PAGE_SIZE.PHOTO_INCREMENT)}
      />

      {previewItem && openIndex !== null && (
        <PhotoPreviewCard
          mode="owned"
          photo={previewItem}
          eventName={event.name}
          index={openIndex + 1}
          total={visibleSlice.length}
          onClose={handleClose}
          onPrev={openIndex > 0 ? handlePrev : undefined}
          onNext={openIndex < visibleSlice.length - 1 ? handleNext : undefined}
          onDownload={handleDownload}
        />
      )}
    </section>
  );
}

function PhotoTile({
  photo,
  index,
  onOpen,
}: {
  photo: PhotographerLibraryPhoto;
  index: number;
  onOpen: () => void;
}) {
  const colorIdx = photo.tone % TONE_COLORS.length;
  const wide = photo.span === "wide";
  const opacity = 0.7 + (index % 3) * 0.1;

  return (
    <div
      className={cn("group relative", wide ? "row-span-1" : "row-span-2")}
      style={{
        animation: `fade-up 0.55s ${Math.min(index * 0.025, 0.6)}s both`,
        opacity: 0,
      }}
    >
      <button
        type="button"
        onClick={onOpen}
        aria-haspopup="dialog"
        aria-label={`Open photo ${index + 1}`}
        className="block w-full h-full text-left rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        <div
          className="relative overflow-hidden rounded-xl h-full transition-transform duration-300 group-hover:-translate-y-[2px]"
          style={{
            backgroundColor: TONE_COLORS[colorIdx],
            opacity,
          }}
        >
          {photo.salesCount > 0 && (
            <span className="absolute bottom-2 right-2 font-mono uppercase tracking-[0.2em] text-[8px] text-bone tnum">
              {photo.salesCount} sold
            </span>
          )}
        </div>
      </button>
    </div>
  );
}
