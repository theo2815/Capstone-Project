"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useCallback, useState } from "react";
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
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import type {
  PhotographerEventSummary,
  PhotographerLibraryPhoto,
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

  const { detail: liveDetail, isMissing } = usePhotographerEventDetail(
    id ?? null,
  );
  const livePhotos = usePhotographerEventPhotos(id ?? null);

  // Only events the photographer actually covers render this page. A 404 means
  // no event_photographer row, or the event is gone — either way the page
  // doesn't exist for them.
  if (isMissing) {
    notFound();
  }

  // Loading: detail is null until the BE responds. Photos load in parallel
  // as an infinite list — the grid shows its own tile skeletons while its
  // first page is in flight.
  if (liveDetail === null) {
    return <FocusedShareSkeleton />;
  }

  // Build a ListEvent-shaped record so the shared sub-components (Hero,
  // ShareHeroBand, etc.) keep their existing prop contracts. Summary fields
  // come straight from the BE; participantCount + status + city are slot
  // fillers manage mode doesn't read.
  const event: ListEvent = {
    id: liveDetail.id,
    slug: liveDetail.slug,
    name: liveDetail.name,
    date: liveDetail.date,
    location: liveDetail.location,
    photoCount: liveDetail.photoCount,
    participantCount: 0,
    status: "ACTIVE",
    state: liveDetail.state as EventState,
    city: liveDetail.location.split(",").pop()?.trim() ?? "",
  };
  const photographer = liveDetail;

  // A covered event with nothing uploaded yet is a normal state, not an error.
  // The events list filters to photoCount > 0, so this is only reachable by
  // deep link or bookmark — and it used to 404, which reads as "your event is
  // gone." ShareHeroBand is deliberately omitted: handing the photographer a
  // public gallery URL to post would send runners to a blank page.
  if (liveDetail.photoCount === 0) {
    return (
      <main className="bg-bone text-ink min-h-screen flex flex-col">
        <SiteHeader />
        <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
          <BackChip />
          <Hero event={event} />
          <NoPhotosYet eventId={event.id} />
        </div>
      </main>
    );
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

// Dashed-panel empty state, matching the /dashboard overview's "No upcoming
// coverage." treatment. The CTA is ink-outline rather than fresh — the Hero's
// StateChip may already own the one fresh accent on this viewport.
function NoPhotosYet({ eventId }: { eventId: string }) {
  return (
    <section className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No photos yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Upload your first batch and this page becomes the public gallery you
        share with runners.
      </p>
      <Link
        href={`${ROUTES.UPLOAD}/${eventId}`}
        className="mt-6 font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors inline-flex items-center justify-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        Upload photos
        <span aria-hidden="true">→</span>
      </Link>
    </section>
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
      className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone mb-8 md:mb-10"
    >
      <span aria-hidden="true">←</span>
      <span>Back to events</span>
    </Link>
  );
}

function Hero({ event }: { event: ListEvent }) {
  return (
    <section className="mb-10 md:mb-12">
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum flex items-center gap-2 flex-wrap">
        <span>{formatLongDate(event.date, true)}</span>
        <span className="text-slate-soft">·</span>
        <StateChip state={event.state} />
      </p>
      <h1 className="font-display text-4xl md:text-6xl font-extrabold tracking-tight text-ink mt-4 leading-[1.05]">
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
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
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
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft shrink-0">
          Share to your followers
        </p>
        <div className="flex items-center gap-2 md:gap-3 flex-wrap">
          <ShareChip
            onClick={() => shareTo("facebook")}
            dotColor="#1877F2"
            label="Facebook"
          />
          {/* Instagram has no web share intent, so this copies the link
              instead of opening a composer. Labelled for what it does — the
              four chips around it really do open a share window, and looking
              identical to them was the whole problem. */}
          <ShareChip
            onClick={handleInstagram}
            dotColor="#E4405F"
            label="Instagram"
            hint="copy"
            ariaLabel="Copy gallery link for Instagram"
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
  hint,
  ariaLabel,
}: {
  onClick: () => void;
  dotColor: string;
  label: string;
  /** Trailing qualifier for chips that don't open a share window. */
  hint?: string;
  ariaLabel?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel ?? `Share to ${label}`}
      className="inline-flex items-center gap-2 font-sans text-sm font-medium border border-line bg-bone hover:bg-bone-deep hover:border-ink py-2 px-4 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      <span
        aria-hidden="true"
        className="size-2 rounded-full shrink-0"
        style={{ backgroundColor: dotColor }}
      />
      <span>{label}</span>
      {hint && (
        <span aria-hidden="true" className="text-slate-soft">
          · {hint}
        </span>
      )}
    </button>
  );
}

function Stats({ photographer }: { photographer: PhotographerEventSummary }) {
  return (
    <section className="mb-10 md:mb-12 flex flex-wrap items-baseline gap-x-8 gap-y-3 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
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
  livePhotos: ReturnType<typeof usePhotographerEventPhotos>;
}) {
  // Flattened server pages loaded so far; Load-more fetches the next page.
  const photos = livePhotos.items;
  // The event's canonical total (event_photographer.photo_count) drives the
  // header stat; the paginator uses the photos endpoint's own total so
  // Load-more reaches every LIVE photo.
  const total = photographer.photoCount;

  const { showToast } = useToast();
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const handleOpen = useCallback((i: number) => setOpenIndex(i), []);
  const handleClose = useCallback(() => setOpenIndex(null), []);
  const handlePrev = useCallback(() => {
    setOpenIndex((i) => (i === null || i === 0 ? i : i - 1));
  }, []);
  // Navigate within the photos loaded so far; the arrow extends as more pages
  // land via Load-more.
  const handleNext = useCallback(() => {
    setOpenIndex((i) => (i === null || i >= photos.length - 1 ? i : i + 1));
  }, [photos.length]);
  const handleDownload = useCallback(async () => {
    const photoId = openIndex !== null ? photos[openIndex]?.id : undefined;
    if (!photoId) return;
    // Signed-URL fetch + programmatic <a download> click (Q-015, 5-min TTL).
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
      // Resolved without a URL (backend 404 / grant missing). Only thrown
      // errors reach the catch, so without this the click is a silent no-op.
      showToast({
        kind: "error",
        message: "Download unavailable — try again.",
      });
    } catch (err) {
      console.error("[photographer/download] failed", err);
      showToast({ kind: "error", message: "Download failed. Try again." });
    }
  }, [openIndex, photos, showToast]);

  const openPhoto = openIndex !== null ? photos[openIndex] : null;
  const previewItem: PhotoPreviewItem | null = openPhoto
    ? {
        id: openPhoto.id,
        bib: openPhoto.bib,
        time: "",
        tone: openPhoto.tone,
        price: 0,
        span: openPhoto.span,
        imageUrl: openPhoto.thumbnailUrl ?? null,
      }
    : null;

  return (
    <section>
      <div className="flex items-baseline justify-between border-b border-line pb-4 mb-6 gap-4">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate tnum">
          Uploaded photos
        </p>
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
          <span className="text-ink">{total.toLocaleString()}</span> total
        </p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
        {/* First photos page loads in parallel with the detail — show tile
            skeletons instead of a silent blank grid under the "N total"
            header. */}
        {livePhotos.isLoading
          ? Array.from({ length: 8 }, (_, i) => (
              <Skeleton key={i} className="h-full w-full rounded-xl" />
            ))
          : photos.map((photo, i) => (
              <PhotoTile
                key={photo.id}
                photo={photo}
                index={i}
                onOpen={() => handleOpen(i)}
              />
            ))}
      </div>

      <LoadMoreButton
        shown={photos.length}
        total={livePhotos.total}
        increment={PAGE_SIZE.PHOTO_INCREMENT}
        onLoadMore={livePhotos.fetchNextPage}
        isLoading={livePhotos.isFetchingNextPage}
      />

      {previewItem && openIndex !== null && (
        <PhotoPreviewCard
          mode="owned"
          photo={previewItem}
          eventName={event.name}
          index={openIndex + 1}
          total={photos.length}
          onClose={handleClose}
          onPrev={openIndex > 0 ? handlePrev : undefined}
          onNext={openIndex < photos.length - 1 ? handleNext : undefined}
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
        {/* Tone-color background acts as a placeholder while the watermarked
            JPEG decodes — once `thumbnailUrl` loads, object-cover fills the
            entire tile and the placeholder is hidden behind it. */}
        <div
          className="relative overflow-hidden rounded-xl h-full transition-transform duration-300 group-hover:-translate-y-[2px]"
          style={{ backgroundColor: TONE_COLORS[colorIdx] }}
        >
          {photo.thumbnailUrl && (
            // eslint-disable-next-line @next/next/no-img-element -- presigned S3 URL; Next/Image would re-proxy and lose the signature.
            <img
              src={photo.thumbnailUrl}
              alt={photo.bib ? `Photo of bib ${photo.bib}` : "Uploaded photo"}
              className="absolute inset-0 w-full h-full object-cover"
              draggable={false}
              loading="lazy"
            />
          )}
          {photo.salesCount > 0 && (
            <span className="absolute bottom-2 right-2 font-mono uppercase tracking-[0.14em] text-[8px] text-bone tnum bg-ink/40 px-1.5 py-0.5 rounded-full backdrop-blur-sm">
              {photo.salesCount} sold
            </span>
          )}
        </div>
      </button>
    </div>
  );
}
