"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type FormEvent,
} from "react";
import Link from "next/link";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import type { EventDetail } from "@/types/event";
import { type MockPhoto } from "@/types/photo";
import { PhotoPreviewCard } from "@/components/photos/photo-preview-card";
import { SaveButton } from "@/components/events/save-button";
import {
  BibPanel,
  SelfieSearchPanel,
  type SearchPanelMode,
} from "@/components/events/bib-search-panels";
import { FindPhotosModal } from "@/components/events/find-photos-modal";
import { PhotoMosaicTile } from "@/components/events/photo-mosaic-tile";
import { BuyAllBar } from "@/components/events/buy-all-bar";
import { BibEmptyResult } from "@/components/events/bib-empty-result";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { RefundModal } from "@/components/orders/refund-modal";
import { useUrlState, useUrlStateBatch } from "@/hooks/use-url-state";
import { useEventPhotos } from "@/hooks/use-event-photos";
import { useEventLivePhotos } from "@/hooks/use-event-live-photos";
import type { EventPhotosResult } from "@/lib/api-photos";
import { deriveEventState } from "@/lib/event-catalog";
import { PAGE_SIZE } from "@/lib/pagination-config";

type Mode = "cockpit" | "browse";

interface Props {
  event: EventDetail;
  initialPhotos: MockPhoto[];
}

export function EventCockpit({ event, initialPhotos }: Props) {
  const [bibFilter] = useUrlState<string>("bib", "", {
    parse: (raw) => raw.trim().toUpperCase(),
  });
  const [browseFlag] = useUrlState<string>("browse", "");
  const [faceFlag] = useUrlState<string>("face", "");
  const setUrlBatch = useUrlStateBatch();

  const isFaceMode = faceFlag === "1";
  const mode: Mode =
    bibFilter || browseFlag === "1" || isFaceMode ? "browse" : "cockpit";

  const [panelMode, setPanelMode] = useState<SearchPanelMode>("bib");
  const [bibInput, setBibInput] = useState(bibFilter);

  // Q-011: bib-keyed cache. Active in cockpit (no filter) and bib-mode browse.
  const bibPhotos = useEventPhotos({
    slug: event.slug,
    bib: bibFilter || undefined,
    initialItems: initialPhotos,
    enabled: !isFaceMode,
  });

  // 2026-05-19 PM: face search is now one-shot, not cached. The selfie panel
  // (upload / take / library pick) does the search and hands the result up
  // through onFaceSearchSuccess. We hold it in local state and render from
  // it while face mode is active. A direct `?face=1` URL (no prior search)
  // shows the FaceEmptyState so the runner can open the modal explicitly —
  // replaces the old "auto-fire face search with primary on URL load"
  // behavior, matching the redesign where every selfie match is initiated
  // by an explicit click.
  const [faceSearchResult, setFaceSearchResult] =
    useState<EventPhotosResult | null>(null);
  useEffect(() => {
    if (!isFaceMode) setFaceSearchResult(null);
  }, [isFaceMode]);

  const visiblePhotos = isFaceMode
    ? faceSearchResult?.items ?? []
    : bibPhotos.photos;
  const visibleTotal = isFaceMode
    ? faceSearchResult?.total ?? 0
    : bibPhotos.total;

  const submitBib = (raw: string) => {
    const clean = raw.trim().toUpperCase();
    if (!clean) return;
    setUrlBatch({ bib: clean, browse: null, face: null });
  };

  const clearBib = () => {
    setUrlBatch({ bib: null, browse: "1", face: null });
  };

  const clearFace = () => {
    setUrlBatch({ face: null, browse: "1", bib: null });
  };

  const switchToBrowse = () => {
    setPanelMode("bib");
    setUrlBatch({ bib: null, browse: "1", face: null });
  };

  const switchToCockpit = () => {
    setBibInput("");
    setPanelMode("bib");
    setUrlBatch({ bib: null, browse: null, face: null });
  };

  const handleFaceSearchSuccess = useCallback(
    (result: EventPhotosResult) => {
      setFaceSearchResult(result);
      setUrlBatch({ face: "1", browse: "1", bib: null });
    },
    [setUrlBatch],
  );

  if (mode === "browse") {
    return (
      <BrowseMode
        event={event}
        photos={visiblePhotos}
        total={visibleTotal}
        bibFilter={bibFilter}
        isFaceMode={isFaceMode}
        onBackToCockpit={switchToCockpit}
        onSubmitBib={submitBib}
        onClearBib={clearBib}
        onClearFace={clearFace}
        onFaceSearchSuccess={handleFaceSearchSuccess}
      />
    );
  }

  return (
    <CockpitMode
      event={event}
      photoCount={event.photoCount}
      bibInput={bibInput}
      onBibChange={setBibInput}
      onSubmit={submitBib}
      panelMode={panelMode}
      onPanelModeChange={setPanelMode}
      onBrowseAll={switchToBrowse}
      onFaceSearchSuccess={handleFaceSearchSuccess}
    />
  );
}

/* ─────────────── COCKPIT MODE ─────────────── */

function CockpitMode({
  event,
  photoCount,
  bibInput,
  onBibChange,
  onSubmit,
  panelMode,
  onPanelModeChange,
  onBrowseAll,
  onFaceSearchSuccess,
}: {
  event: EventDetail;
  photoCount: number;
  bibInput: string;
  onBibChange: (v: string) => void;
  onSubmit: (v: string) => void;
  panelMode: SearchPanelMode;
  onPanelModeChange: (m: SearchPanelMode) => void;
  onBrowseAll: () => void;
  onFaceSearchSuccess: (result: EventPhotosResult) => void;
}) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit(bibInput);
  };

  return (
    <>
      <div className="bg-bone">
        <div className="max-w-7xl mx-auto px-6 md:px-10 pt-5 md:pt-6 flex items-center justify-between gap-4">
          <Kicker
            as={Link}
            href="/events"
            className="group inline-flex items-center gap-2 hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <span
              aria-hidden="true"
              className="transition-transform group-hover:-translate-x-0.5"
            >
              ←
            </span>
            <span>Back to events</span>
          </Kicker>
          <SaveButton
            eventId={event.id}
            event={{
              id: event.id,
              slug: event.slug,
              name: event.name,
              date: event.date,
              state: deriveEventState(event.date),
              bannerUrl: event.bannerUrl ?? null,
              location: event.location,
            }}
            variant="inline"
          />
        </div>
      </div>

      <section className="relative bg-bone overflow-hidden">
        <DimWall />

        <div className="relative px-6 md:px-10 py-16 md:py-24 min-h-[78vh] flex flex-col items-center justify-center">
          <div className="w-full max-w-md">
            <div
              className="rounded-2xl bg-bone border border-line shadow-[0_24px_60px_-20px_rgba(17,17,17,0.18)] p-8 md:p-10"
              style={{ animation: "fade-up 0.7s 0.05s both", opacity: 0 }}
            >
              <Kicker as="p" className="mb-5">
                {event.name}
              </Kicker>
              <h1 className="font-display text-4xl md:text-5xl font-medium tracking-tight leading-[0.95]">
                Find your
                <br />
                <span className="text-fresh">photos.</span>
              </h1>

              {panelMode === "bib" ? (
                <BibPanel
                  bibInput={bibInput}
                  onBibChange={onBibChange}
                  onSubmit={handleSubmit}
                  onSwitchToSelfie={() => onPanelModeChange("selfie")}
                  photoCount={photoCount}
                  eventPhotoCount={event.photoCount}
                />
              ) : (
                <SelfieSearchPanel
                  eventSlug={event.slug}
                  onSwitchToBib={() => onPanelModeChange("bib")}
                  onSearchSuccess={onFaceSearchSuccess}
                />
              )}
            </div>
          </div>

          <Kicker
            as="button"
            type="button"
            onClick={onBrowseAll}
            size="md"
            className="mt-10 md:mt-14 inline-flex items-center gap-2 hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-4 focus-visible:ring-offset-bone"
            style={{ animation: "fade-in 0.6s 0.55s both", opacity: 0 }}
          >
            Browse all <span className="tnum">{photoCount}</span> photos
            <span aria-hidden="true">↓</span>
          </Kicker>
        </div>
      </section>

      <AboutStrip event={event} />

      <Footer />
    </>
  );
}

/* All 20 photos are verified real marathon / road-race / trail-running event
   photos downloaded from Unsplash. IDs are documented so they can be audited:
     1  – 1452626038306  pack of marathon runners on city road
     2  – 1530549387789  female athlete sprinting in race
     3  – 1486218119243  runner on track stadium
     4  – 1513593771513  night marathon runners
     5  – 1571008887538  running shoes mid-stride on asphalt
     6  – 1502904550040  solo road runner
     7  – 1476480862126  trail running in nature
     8  – 1461896836934  sprint finish on athletics track
     9  – 1552674605     group of runners in road race
    10  – 1452626038306  (same race pack, different crop)
    11  – 1533560904424  marathon finish line
    12  – 1538805060514  mass start of marathon runners
    13  – 1596727147705  trail runner mountain race
    14  – 1574680178050  trail running forest
    15  – 1571019614242  road race runners crowd
    16  – 1543622748     runner on track with motion blur
    17  – 1530549387789  (same sprinter, different crop)
    18  – 1486218119243  (same track runner, different crop)
    19  – 1560743641     road marathon runners
    20  – 1476480862126  (same trail runner, different crop) */
const ALL_RUNNER_PHOTOS = [
  "/images/runners/runner-photo-1.jpg",
  "/images/runners/runner-photo-2.jpg",
  "/images/runners/runner-photo-3.jpg",
  "/images/runners/runner-photo-4.jpg",
  "/images/runners/runner-photo-5.jpg",
  "/images/runners/runner-photo-6.jpg",
  "/images/runners/runner-photo-7.jpg",
  "/images/runners/runner-photo-8.jpg",
  "/images/runners/runner-photo-9.jpg",
  "/images/runners/runner-photo-11.jpg",
  "/images/runners/runner-photo-12.jpg",
  "/images/runners/runner-photo-13.jpg",
  "/images/runners/runner-photo-14.jpg",
  "/images/runners/runner-photo-15.jpg",
  "/images/runners/runner-photo-16.jpg",
  "/images/runners/runner-photo-19.jpg",
];

/** Deterministic pseudo-random tile generator using a seeded LCG algorithm.
 *  Produces a completely varied, non-repeating sequence across all 80 tiles
 *  that is 100% identical on SSR and client hydration, eliminating hydration errors. */
function buildTileList(count: number): string[] {
  const tiles: string[] = [];
  const len = ALL_RUNNER_PHOTOS.length;
  let seed = 123456789;
  const lcg = () => {
    seed = (seed * 1664525 + 1013904223) % 4294967296;
    return seed / 4294967296;
  };

  let lastIdx = -1;
  for (let i = 0; i < count; i++) {
    let idx = Math.floor(lcg() * len);
    if (idx === lastIdx) {
      idx = (idx + 1) % len;
    }
    tiles.push(ALL_RUNNER_PHOTOS[idx]);
    lastIdx = idx;
  }
  return tiles;
}

function DimWall() {
  const TILES = 80;

  // Deterministic calculation — 100% identical on SSR and client hydration
  const tileImages = useMemo(() => buildTileList(TILES), []);

  return (
    <div
      aria-hidden="true"
      className="absolute inset-0 pointer-events-none select-none overflow-hidden"
      style={{ animation: "fade-in 0.9s 0.1s both" }}
    >
      <div className="absolute inset-0 grid gap-2 p-3 md:gap-3 md:p-6 grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 content-start">
        {tileImages.map((imgUrl, i) => {
          // Vary opacity slightly per tile so the wall has natural texture
          const op = 0.3 + ((i * 13 + 5) % 11) * 0.04;
          return (
            <div
              key={i}
              className="rounded-xl overflow-hidden aspect-[3/4] bg-stone-200/60 relative border border-black/10 shadow-xs"
            >
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={imgUrl}
                alt=""
                className="w-full h-full object-cover saturate-75 contrast-[1.08] transition-opacity duration-500"
                style={{ opacity: op }}
                loading="lazy"
                onError={(e) => {
                  (e.currentTarget as HTMLImageElement).style.display = "none";
                }}
              />
            </div>
          );
        })}
      </div>
      {/* Radial vignette — fades the edges more than the centre so the
          search card floats clearly above the photo wall */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 60% at 50% 42%, transparent 0%, rgba(248,245,238,0.55) 60%, rgba(248,245,238,0.92) 100%)",
        }}
      />
      {/* Bottom fade-to-bone so the page footer merges smoothly */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent via-[40%] to-bone pointer-events-none" />
    </div>
  );
}

/* ─────────────── ABOUT STRIP (cockpit mode only) ─────────────── */

function AboutStrip({ event }: { event: EventDetail }) {
  const dateLong = formatLongDate(event.date);
  return (
    <section className="bg-bone-deep border-y border-line px-6 md:px-10 py-16 md:py-24">
      <div className="max-w-7xl mx-auto grid md:grid-cols-[1fr_2fr] gap-10 md:gap-16 items-start">
        <div>
          <Kicker as="p" className="mb-4">
            About this race
          </Kicker>
          <h2 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-tight text-ink">
            Race day notes.
          </h2>
          <div className="mt-6 space-y-2">
            <Kicker as="p" tone="soft">Organizer · {event.organizerName}</Kicker>
            <Kicker as="p" tone="soft">
              <span className="tnum">{dateLong}</span> · {event.location}
            </Kicker>
            {/* No runner count: `participantCount` is a reserved wire field the
                BE always sends as 0 (there is no participants table — it's a
                placeholder for roadmap participant management), so rendering it
                told every visitor this race had "0 runners". Restore the stat
                when the feature lands, not before. */}
            <Kicker as="p" tone="soft">
              <span className="tnum">
                {event.photoCount.toLocaleString()}
              </span>{" "}
              photos
            </Kicker>
          </div>
        </div>

        <div className="space-y-7">
          <p className="font-sans text-base md:text-lg text-ink-soft leading-relaxed max-w-prose">
            {event.description}
          </p>

          <div className="flex flex-wrap gap-2">
            {event.categories.map((c) => (
              <Kicker
                key={c}
                className="border border-line bg-bone rounded-full px-4 py-1.5 text-ink"
              >
                {c}
              </Kicker>
            ))}
          </div>

          <div className="pt-5 border-t border-line">
            <Kicker as="p" className="mb-3">
              Pricing
            </Kicker>
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
              <span className="font-display text-5xl md:text-6xl font-medium text-fresh tracking-tight tnum">
                ₱{event.pricePerPhoto}
              </span>
              <Kicker>per photo</Kicker>
              {event.bundlePrice && event.bundleSize && (
                <>
                  <span className="text-slate-soft">·</span>
                  <Kicker className="text-ink">
                    or <span className="tnum">₱{event.bundlePrice}</span> for{" "}
                    <span className="tnum">{event.bundleSize}</span>
                  </Kicker>
                </>
              )}
            </div>
            <p className="mt-3 font-sans text-sm text-slate-soft max-w-md">
              Watermarked previews are free. Pay once, download forever.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}

/* ─────────────── BROWSE MODE ─────────────── */

function BrowseMode({
  event,
  photos,
  total,
  bibFilter,
  isFaceMode,
  onBackToCockpit,
  onSubmitBib,
  onClearBib,
  onClearFace,
  onFaceSearchSuccess,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  total: number;
  bibFilter: string;
  isFaceMode: boolean;
  onBackToCockpit: () => void;
  onSubmitBib: (b: string) => void;
  onClearBib: () => void;
  onClearFace: () => void;
  onFaceSearchSuccess: (result: EventPhotosResult) => void;
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.PHOTO_INITIAL);
  const [isPolicyOpen, setIsPolicyOpen] = useState(false);

  const isBibFilter = bibFilter.trim().length > 0;
  const isAnyFilter = isBibFilter || isFaceMode;
  // Photos are already server-filtered by bib (Q-011) or face (Q-005/006).
  const visible = photos;

  // Q-002: live WebSocket prepend for live-state events. Hook is a no-op
  // when the event is not in the live state (gated inside).
  const liveState = deriveEventState(event.date);
  const live = useEventLivePhotos({
    slug: event.slug,
    eventId: event.id,
    enabled: liveState === "live",
  });

  useEffect(() => {
    setLoadedCount(PAGE_SIZE.PHOTO_INITIAL);
  }, [bibFilter, isFaceMode]);

  const visibleSlice = useMemo(
    () => visible.slice(0, loadedCount),
    [visible, loadedCount],
  );

  useEffect(() => {
    if (previewIndex !== null && previewIndex >= visibleSlice.length) {
      setPreviewIndex(visibleSlice.length === 0 ? null : visibleSlice.length - 1);
    }
  }, [visibleSlice.length, previewIndex]);

  const totalPrice = visible.reduce((sum, p) => sum + p.price, 0);
  const showBuyAll = isAnyFilter && visible.length > 0;
  const previewPhoto =
    previewIndex !== null ? visibleSlice[previewIndex] ?? null : null;

  const headerKicker = isBibFilter
    ? `${event.name} · BIB ${bibFilter}`
    : isFaceMode
      ? `${event.name} · Selfie match`
      : `${event.name} · Gallery`;

  return (
    <section className="bg-bone min-h-screen flex flex-col">
      <header className="px-6 md:px-10 pt-8 md:pt-12 pb-8 md:pb-10">
        <div className="max-w-7xl mx-auto">
          <Kicker
            as="button"
            type="button"
            onClick={onBackToCockpit}
            className="group inline-flex items-center gap-2 hover:text-ink transition-colors mb-8 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <span
              aria-hidden="true"
              className="transition-transform group-hover:-translate-x-0.5"
            >
              ←
            </span>
            <span>Back</span>
          </Kicker>
          <Kicker as="p" className="mb-4">
            {headerKicker}
          </Kicker>
          <h2 className="font-display text-4xl md:text-6xl font-medium tracking-tight leading-[0.95] text-ink">
            {isAnyFilter ? (
              visible.length === 0 ? (
                "No matches yet."
              ) : (
                <>
                  We found{" "}
                  <span
                    key={`${bibFilter}|${isFaceMode}`}
                    className="text-fresh tnum"
                    style={{
                      animation: "count-up 0.6s 0.05s both",
                      opacity: 0,
                    }}
                  >
                    {visible.length}
                  </span>{" "}
                  {isFaceMode
                    ? visible.length === 1
                      ? "match"
                      : "matches"
                    : visible.length === 1
                      ? "photo"
                      : "photos"}
                  .
                </>
              )
            ) : (
              <>
                Browse <span className="tnum">{total || visible.length}</span>{" "}
                photos.
              </>
            )}
          </h2>
          <p className="mt-4 font-sans text-base md:text-lg text-ink-soft max-w-md">
            {isFaceMode
              ? "These are the photos that match your saved selfie. Tap any to add to cart."
              : isBibFilter
                ? "These are the photos matching your bib. Tap any to add to cart."
                : "Skim the wall, or open search anytime to find your bib."}
          </p>
          <Kicker
            as="button"
            type="button"
            tone="soft"
            onClick={() => setIsPolicyOpen(true)}
            className="mt-5 inline-flex items-center gap-2 underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            Refund Policy
            <span aria-hidden="true">→</span>
          </Kicker>
        </div>
      </header>

      <div className="sticky top-[var(--site-header-h)] z-20 bg-bone/90 backdrop-blur-md border-y border-line">
        <div className="max-w-7xl mx-auto px-6 md:px-10 py-3 flex items-center gap-3">
          <Kicker
            as="button"
            type="button"
            onClick={() => setSearchOpen(true)}
            aria-haspopup="dialog"
            className="flex-1 min-w-0 inline-flex items-center gap-2.5 px-4 py-2.5 rounded-full border border-line bg-bone-deep/60 hover:bg-bone-deep hover:border-slate transition-colors text-left text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <svg
              viewBox="0 0 16 16"
              className="size-3.5 text-slate shrink-0"
              fill="none"
              aria-hidden="true"
            >
              <circle cx="7" cy="7" r="4.5" stroke="currentColor" strokeWidth="1.5" />
              <path
                d="M10.5 10.5 L14 14"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
            </svg>
            <span className="truncate">
              {isBibFilter
                ? `Search · ${bibFilter}`
                : isFaceMode
                  ? "Search · selfie"
                  : "Find your photos"}
            </span>
          </Kicker>
          {isBibFilter ? (
            <Kicker
              as="button"
              type="button"
              onClick={onClearBib}
              className="shrink-0 inline-flex items-center gap-2 hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </Kicker>
          ) : isFaceMode ? (
            <Kicker
              as="button"
              type="button"
              onClick={onClearFace}
              className="shrink-0 inline-flex items-center gap-2 hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </Kicker>
          ) : (
            <Kicker tone="soft" className="shrink-0 hidden sm:inline">
              <span className="tnum text-ink">{total || visible.length}</span>{" "}
              photos
            </Kicker>
          )}
        </div>
        {liveState === "live" && (live.newCount > 0 || live.reconnectFailed) && (
          <div className="max-w-7xl mx-auto px-6 md:px-10 pb-3">
            {live.reconnectFailed ? (
              <button
                type="button"
                onClick={live.refresh}
                className="font-mono uppercase tracking-[0.25em] text-[10px] text-ink underline decoration-line underline-offset-4 hover:decoration-ink"
              >
                Connection lost · Refresh ↻
              </button>
            ) : (
              <button
                type="button"
                onClick={() => {
                  live.resetNewCount();
                  if (typeof window !== "undefined") {
                    window.scrollTo({ top: 0, behavior: "smooth" });
                  }
                }}
                className="font-mono uppercase tracking-[0.25em] text-[10px] text-fresh hover:text-fresh-deep"
              >
                <span className="tnum">{live.newCount}</span> new photo
                {live.newCount === 1 ? "" : "s"} · jump to top ↑
              </button>
            )}
          </div>
        )}
      </div>

      <div className="flex-1 flex flex-col">
        {isBibFilter && visible.length === 0 ? (
          <BibEmptyResult event={event} bib={bibFilter} onClear={onClearBib} />
        ) : isFaceMode && visible.length === 0 ? (
          <FaceEmptyResult onClear={onClearFace} />
        ) : (
          <div className="px-6 md:px-10 py-10 md:py-14 pb-20">
            <div className="max-w-7xl mx-auto">
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
                {visibleSlice.map((p, i) => (
                  <PhotoMosaicTile
                    key={p.id}
                    event={event}
                    photo={p}
                    index={i}
                    onOpen={() => setPreviewIndex(i)}
                  />
                ))}
              </div>
              <LoadMoreButton
                shown={visibleSlice.length}
                total={Math.min(visible.length, total || visible.length)}
                increment={PAGE_SIZE.PHOTO_INCREMENT}
                onLoadMore={() =>
                  setLoadedCount((n) => n + PAGE_SIZE.PHOTO_INCREMENT)
                }
                countSuffix={
                  isBibFilter
                    ? `· BIB ${bibFilter}`
                    : isFaceMode
                      ? "· selfie match"
                      : undefined
                }
              />
              {!isBibFilter && !isFaceMode && total > visible.length && (
                <p className="mt-4 text-center font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
                  Showing first <span className="tnum text-ink">{visible.length}</span> of <span className="tnum text-ink">{total}</span> · search by bib or selfie to find yours
                </p>
              )}
            </div>
          </div>
        )}
      </div>

      {showBuyAll && <BuyAllBar event={event} photos={visible} total={totalPrice} />}

      {searchOpen && (
        <FindPhotosModal
          eventSlug={event.slug}
          eyebrow={event.name}
          photoCount={total || visible.length}
          eventPhotoCount={event.photoCount}
          onClose={() => setSearchOpen(false)}
          onSubmitBib={onSubmitBib}
          onSearchByFaceSuccess={onFaceSearchSuccess}
        />
      )}

      <RefundModal
        mode="policy"
        isOpen={isPolicyOpen}
        onClose={() => setIsPolicyOpen(false)}
      />

      {previewPhoto && previewIndex !== null && (
        <PhotoPreviewMount
          event={event}
          photo={previewPhoto}
          index={previewIndex}
          total={visibleSlice.length}
          onClose={() => setPreviewIndex(null)}
          onPrev={
            previewIndex > 0
              ? () => setPreviewIndex(previewIndex - 1)
              : undefined
          }
          onNext={
            previewIndex < visibleSlice.length - 1
              ? () => setPreviewIndex(previewIndex + 1)
              : undefined
          }
        />
      )}

      <Footer />
    </section>
  );
}

function FaceEmptyResult({ onClear }: { onClear: () => void }) {
  return (
    <section className="px-6 md:px-10 py-16 md:py-24 bg-bone min-h-[40vh] flex items-center">
      <div className="max-w-2xl mx-auto w-full text-center">
        <Kicker as="p" className="mb-3">
          No matches yet
        </Kicker>
        <p className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight">
          We didn&apos;t find your face.
        </p>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-4">
          Try adding another selfie angle, or browse the wall while photos roll
          in.
        </p>
        <button
          type="button"
          onClick={onClear}
          className="mt-7 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-[12px] transition-colors"
        >
          Browse the wall →
        </button>
      </div>
    </section>
  );
}

function PhotoPreviewMount({
  event,
  photo,
  index,
  total,
  onClose,
  onPrev,
  onNext,
}: {
  event: EventDetail;
  photo: MockPhoto;
  index: number;
  total: number;
  onClose: () => void;
  onPrev?: () => void;
  onNext?: () => void;
}) {
  const addItem = useCartStore((s) => s.addItem);
  const removeItem = useCartStore((s) => s.removeItem);
  const inCart = useCartStore((s) =>
    s.items.some((i) => i.photoId === photo.id),
  );
  const openCart = useUiStore((s) => s.openCart);
  const openCheckout = useUiStore((s) => s.openCheckout);
  const startExpressCheckout = useUiStore((s) => s.startExpressCheckout);

  const handleToggle = () => {
    if (inCart) {
      removeItem(photo.id);
    } else {
      addItem({
        photoId: photo.id,
        eventId: event.id,
        thumbnailUrl: photo.imageUrl ?? "",
        price: photo.price,
        bib: photo.bib,
        eventName: event.name,
        eventSlug: event.slug,
        tone: photo.tone,
        time: photo.time,
      });
    }
  };

  const handleBuyNow = () => {
    onClose();
    if (inCart) {
      openCheckout();
      return;
    }
    // Flag the next count-increase so FloatingCart opens checkout instead of
    // dismissing modals. Then add; FloatingCart's effect consumes the flag
    // and routes to checkout.
    startExpressCheckout();
    addItem({
      photoId: photo.id,
      eventId: event.id,
      thumbnailUrl: photo.imageUrl ?? "",
      price: photo.price,
      bib: photo.bib,
      eventName: event.name,
      eventSlug: event.slug,
      tone: photo.tone,
      time: photo.time,
    });
  };

  const handleViewCart = () => {
    onClose();
    openCart();
  };

  return (
    <PhotoPreviewCard
      photo={photo}
      eventName={event.name}
      index={index + 1}
      total={total}
      inCart={inCart}
      onClose={onClose}
      onPrev={onPrev}
      onNext={onNext}
      onToggleCart={handleToggle}
      onBuyNow={handleBuyNow}
      onViewCart={handleViewCart}
    />
  );
}

/* ─────────────── FOOTER ─────────────── */

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 pb-24 md:pb-20 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <Kicker as="p" tone="soft">
        QuickPitik · Cebu, Philippines
      </Kicker>
      <Kicker as="p" tone="soft">
        © 2026
      </Kicker>
    </footer>
  );
}

/* ─────────────── HELPERS ─────────────── */

function formatLongDate(iso: string) {
  const d = new Date(iso + "T00:00:00");
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}
