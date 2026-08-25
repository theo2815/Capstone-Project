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
import { cn } from "@/lib/utils";
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
              className="rounded-2xl bg-surface border border-line shadow-[var(--shadow-lift)] p-8 md:p-10"
              style={{ animation: "fade-up 0.7s 0.05s both", opacity: 0 }}
            >
              <Kicker as="p" className="mb-5">
                {event.name}
              </Kicker>
              <h1 className="font-hero text-ink text-5xl md:text-6xl">
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

function DimWall() {
  const TILES = 80;
  return (
    <div
      aria-hidden="true"
      className="absolute inset-0 pointer-events-none select-none overflow-hidden"
      style={{ animation: "fade-in 0.9s 0.1s both" }}
    >
      <div className="absolute inset-0 grid gap-2 p-3 md:gap-3 md:p-6 grid-cols-4 sm:grid-cols-6 md:grid-cols-8 lg:grid-cols-10 content-start">
        {Array.from({ length: TILES }).map((_, i) => {
          const op = 0.05 + ((i * 17) % 11) * 0.009;
          return (
            <div
              key={i}
              className="rounded-xl bg-ink aspect-[3/4]"
              style={{ opacity: op }}
            />
          );
        })}
      </div>
      <div className="absolute inset-0 bg-gradient-to-b from-bone/0 via-bone/55 via-[55%] to-bone pointer-events-none" />
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
          <h2 className="font-display font-extrabold text-3xl md:text-4xl tracking-tight leading-tight text-ink">
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
              <span className="font-display text-5xl md:text-6xl font-extrabold text-fresh tracking-tight tnum">
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
          <h2 className="font-hero text-ink text-4xl md:text-6xl">
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
          <button
            type="button"
            onClick={() => setSearchOpen(true)}
            aria-haspopup="dialog"
            className="flex-1 min-w-0 inline-flex items-center gap-2.5 px-4 py-2.5 rounded-full border border-line-strong bg-surface shadow-[var(--shadow-card)] hover:border-ink transition-colors text-left font-sans text-sm font-medium text-slate hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <svg
              viewBox="0 0 16 16"
              className="size-4 text-slate shrink-0"
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
            <span className={cn("truncate", isAnyFilter && "text-ink font-bold")}>
              {isBibFilter
                ? `Search · ${bibFilter}`
                : isFaceMode
                  ? "Search · selfie"
                  : "Find your photos"}
            </span>
          </button>
          {isBibFilter ? (
            <button
              type="button"
              onClick={onClearBib}
              className="shrink-0 inline-flex items-center gap-2 rounded-full border border-line px-3.5 py-2 font-sans text-sm font-medium text-ink hover:border-ink hover:bg-bone-deep transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </button>
          ) : isFaceMode ? (
            <button
              type="button"
              onClick={onClearFace}
              className="shrink-0 inline-flex items-center gap-2 rounded-full border border-line px-3.5 py-2 font-sans text-sm font-medium text-ink hover:border-ink hover:bg-bone-deep transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </button>
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
                className="font-mono uppercase tracking-[0.25em] text-[12px] text-ink underline decoration-line underline-offset-4 hover:decoration-ink"
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
                className="font-mono uppercase tracking-[0.25em] text-[12px] text-fresh hover:text-fresh-deep underline decoration-fresh/40 underline-offset-4 hover:decoration-fresh-deep"
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
                <p className="mt-4 text-center font-mono uppercase tracking-[0.25em] text-[12px] text-slate-soft">
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
        <p className="font-display font-extrabold text-3xl md:text-4xl text-ink tracking-tight">
          We didn&apos;t find your face.
        </p>
        <p className="font-sans text-base md:text-lg text-ink-soft mt-4">
          Try adding another selfie angle, or browse the wall while photos roll
          in.
        </p>
        <button
          type="button"
          onClick={onClear}
          className="mt-7 inline-flex items-center gap-2 bg-fresh hover:bg-fresh-deep text-surface px-6 py-3 rounded-full font-display font-bold text-[15px] transition-colors"
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
