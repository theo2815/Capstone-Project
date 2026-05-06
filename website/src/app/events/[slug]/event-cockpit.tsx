"use client";

import {
  useState,
  useMemo,
  useEffect,
  type FormEvent,
} from "react";
import Link from "next/link";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import type { EventDetail } from "@/types/event";
import { type MockPhoto } from "./mock-photos";
import { PhotoPreviewCard } from "@/components/photos/photo-preview-card";
import { SaveButton } from "@/components/events/save-button";
import {
  BibPanel,
  SelfiePendingPanel,
  type SearchPanelMode,
} from "@/components/events/bib-search-panels";
import { FindPhotosModal } from "@/components/events/find-photos-modal";
import { PhotoMosaicTile } from "@/components/events/photo-mosaic-tile";
import { BuyAllBar } from "@/components/events/buy-all-bar";
import { BibEmptyResult } from "@/components/events/bib-empty-result";

type Mode = "cockpit" | "browse";

interface Props {
  event: EventDetail;
  photos: MockPhoto[];
}

export function EventCockpit({ event, photos }: Props) {
  const sp = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const initialBibParam = (sp.get("bib") ?? "").trim().toUpperCase();
  const initialBrowse = sp.get("browse") === "1";
  const initialMode: Mode =
    initialBrowse || initialBibParam ? "browse" : "cockpit";

  const [mode, setMode] = useState<Mode>(initialMode);
  const [panelMode, setPanelMode] = useState<SearchPanelMode>("bib");
  const [bibInput, setBibInput] = useState(initialBibParam);
  const [bibFilter, setBibFilter] = useState<string>(initialBibParam);

  const replaceUrl = (next: { bib?: string; browse?: boolean } | null) => {
    if (!next) {
      router.replace(pathname, { scroll: false });
      return;
    }
    const params = new URLSearchParams();
    if (next.bib) {
      params.set("bib", next.bib);
    } else if (next.browse) {
      params.set("browse", "1");
    }
    const qs = params.toString();
    router.replace(qs ? `${pathname}?${qs}` : pathname, { scroll: false });
  };

  const submitBib = (raw: string) => {
    const clean = raw.trim().toUpperCase();
    if (!clean) return;
    setBibFilter(clean);
    setMode("browse");
    replaceUrl({ bib: clean });
  };

  const clearBib = () => {
    setBibFilter("");
    replaceUrl({ browse: true });
  };

  const switchToBrowse = () => {
    setMode("browse");
    setBibFilter("");
    setPanelMode("bib");
    replaceUrl({ browse: true });
  };

  const switchToCockpit = () => {
    setMode("cockpit");
    setBibInput("");
    setBibFilter("");
    setPanelMode("bib");
    replaceUrl(null);
  };

  if (mode === "browse") {
    return (
      <BrowseMode
        event={event}
        photos={photos}
        bibFilter={bibFilter}
        onBackToCockpit={switchToCockpit}
        onSubmitBib={submitBib}
        onClearBib={clearBib}
      />
    );
  }

  return (
    <CockpitMode
      event={event}
      photos={photos}
      bibInput={bibInput}
      onBibChange={setBibInput}
      onSubmit={submitBib}
      panelMode={panelMode}
      onPanelModeChange={setPanelMode}
      onBrowseAll={switchToBrowse}
    />
  );
}

/* ─────────────── COCKPIT MODE ─────────────── */

function CockpitMode({
  event,
  photos,
  bibInput,
  onBibChange,
  onSubmit,
  panelMode,
  onPanelModeChange,
  onBrowseAll,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  bibInput: string;
  onBibChange: (v: string) => void;
  onSubmit: (v: string) => void;
  panelMode: SearchPanelMode;
  onPanelModeChange: (m: SearchPanelMode) => void;
  onBrowseAll: () => void;
}) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit(bibInput);
  };

  return (
    <>
      <div className="bg-bone">
        <div className="max-w-7xl mx-auto px-6 md:px-10 pt-5 md:pt-6 flex items-center justify-between gap-4">
          <Link
            href="/events"
            className="group inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[10px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <span
              aria-hidden="true"
              className="transition-transform group-hover:-translate-x-0.5"
            >
              ←
            </span>
            <span>Back to events</span>
          </Link>
          <SaveButton eventId={event.id} variant="inline" />
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
              <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-5">
                {event.name}
              </p>
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
                  photoCount={photos.length}
                  eventPhotoCount={event.photoCount}
                />
              ) : (
                <SelfiePendingPanel
                  onSwitchToBib={() => onPanelModeChange("bib")}
                />
              )}
            </div>
          </div>

          <button
            type="button"
            onClick={onBrowseAll}
            className="mt-10 md:mt-14 inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[11px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-4 focus-visible:ring-offset-bone"
            style={{ animation: "fade-in 0.6s 0.55s both", opacity: 0 }}
          >
            Browse all <span className="tnum">{photos.length}</span> photos
            <span aria-hidden="true">↓</span>
          </button>
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
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-4">
            About this race
          </p>
          <h2 className="font-display text-3xl md:text-4xl font-medium tracking-tight leading-tight text-ink">
            Race day notes.
          </h2>
          <div className="mt-6 space-y-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
            <p>Organizer · {event.organizerName}</p>
            <p>
              <span className="tnum">{dateLong}</span> · {event.location}
            </p>
            <p>
              <span className="tnum">
                {event.photoCount.toLocaleString()}
              </span>{" "}
              photos ·{" "}
              <span className="tnum">
                {event.participantCount.toLocaleString()}
              </span>{" "}
              runners
            </p>
          </div>
        </div>

        <div className="space-y-7">
          <p className="font-sans text-base md:text-lg text-ink-soft leading-relaxed max-w-prose">
            {event.description}
          </p>

          <div className="flex flex-wrap gap-2">
            {event.categories.map((c) => (
              <span
                key={c}
                className="border border-line bg-bone rounded-full px-4 py-1.5 font-mono uppercase tracking-[0.25em] text-[10px] text-ink"
              >
                {c}
              </span>
            ))}
          </div>

          <div className="pt-5 border-t border-line">
            <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-3">
              Pricing
            </p>
            <div className="flex flex-wrap items-baseline gap-x-4 gap-y-2">
              <span className="font-display text-5xl md:text-6xl font-medium text-fresh tracking-tight tnum">
                ₱{event.pricePerPhoto}
              </span>
              <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate">
                per photo
              </span>
              {event.bundlePrice && event.bundleSize && (
                <>
                  <span className="text-slate-soft">·</span>
                  <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-ink">
                    or <span className="tnum">₱{event.bundlePrice}</span> for{" "}
                    <span className="tnum">{event.bundleSize}</span>
                  </span>
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
  bibFilter,
  onBackToCockpit,
  onSubmitBib,
  onClearBib,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  bibFilter: string;
  onBackToCockpit: () => void;
  onSubmitBib: (b: string) => void;
  onClearBib: () => void;
}) {
  const [searchOpen, setSearchOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);

  const cleanedQuery = bibFilter.replace(/^B-/i, "").trim().toUpperCase();
  const isFiltered = cleanedQuery.length > 0;

  const visible = useMemo(() => {
    if (!isFiltered) return photos;
    return photos.filter((p) => {
      if (!p.bib) return false;
      const num = p.bib.replace(/^B-/, "");
      return num.includes(cleanedQuery);
    });
  }, [photos, cleanedQuery, isFiltered]);

  useEffect(() => {
    if (previewIndex !== null && previewIndex >= visible.length) {
      setPreviewIndex(visible.length === 0 ? null : visible.length - 1);
    }
  }, [visible.length, previewIndex]);

  const total = visible.reduce((sum, p) => sum + p.price, 0);
  const showBuyAll = isFiltered && visible.length > 0;
  const previewPhoto =
    previewIndex !== null ? visible[previewIndex] ?? null : null;

  return (
    <section className="bg-bone min-h-screen flex flex-col">
      <header className="px-6 md:px-10 pt-8 md:pt-12 pb-8 md:pb-10">
        <div className="max-w-7xl mx-auto">
          <button
            type="button"
            onClick={onBackToCockpit}
            className="group inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[10px] text-slate hover:text-ink transition-colors mb-8 rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <span
              aria-hidden="true"
              className="transition-transform group-hover:-translate-x-0.5"
            >
              ←
            </span>
            <span>Back</span>
          </button>
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-4">
            {isFiltered
              ? `${event.name} · BIB ${bibFilter}`
              : `${event.name} · Gallery`}
          </p>
          <h2 className="font-display text-4xl md:text-6xl font-medium tracking-tight leading-[0.95] text-ink">
            {isFiltered ? (
              visible.length === 0 ? (
                "No matches yet."
              ) : (
                <>
                  We found{" "}
                  <span
                    key={cleanedQuery}
                    className="text-fresh tnum"
                    style={{
                      animation: "count-up 0.6s 0.05s both",
                      opacity: 0,
                    }}
                  >
                    {visible.length}
                  </span>{" "}
                  {visible.length === 1 ? "photo" : "photos"}.
                </>
              )
            ) : (
              <>
                Browse <span className="tnum">{photos.length}</span> photos.
              </>
            )}
          </h2>
          <p className="mt-4 font-sans text-base md:text-lg text-ink-soft max-w-md">
            {isFiltered
              ? "These are the photos matching your bib. Tap any to add to cart."
              : "Skim the wall, or open search anytime to find your bib."}
          </p>
        </div>
      </header>

      <div className="sticky top-[3.75rem] z-20 bg-bone/90 backdrop-blur-md border-y border-line">
        <div className="max-w-7xl mx-auto px-6 md:px-10 py-3 flex items-center gap-3">
          <button
            type="button"
            onClick={() => setSearchOpen(true)}
            aria-haspopup="dialog"
            className="flex-1 min-w-0 inline-flex items-center gap-2.5 px-4 py-2.5 rounded-full border border-line bg-bone-deep/60 hover:bg-bone-deep hover:border-slate transition-colors text-left font-mono uppercase tracking-[0.25em] text-[10px] text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
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
              {isFiltered ? `Search · ${bibFilter}` : "Find your photos"}
            </span>
          </button>
          {isFiltered ? (
            <button
              type="button"
              onClick={onClearBib}
              className="shrink-0 inline-flex items-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </button>
          ) : (
            <span className="shrink-0 font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft hidden sm:inline">
              <span className="tnum text-ink">{photos.length}</span> photos
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {isFiltered && visible.length === 0 ? (
          <BibEmptyResult event={event} bib={bibFilter} onClear={onClearBib} />
        ) : (
          <div className="px-6 md:px-10 py-10 md:py-14 pb-20">
            <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
              {visible.map((p, i) => (
                <PhotoMosaicTile
                  key={p.id}
                  event={event}
                  photo={p}
                  index={i}
                  onOpen={() => setPreviewIndex(i)}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {showBuyAll && <BuyAllBar event={event} photos={visible} total={total} />}

      {searchOpen && (
        <FindPhotosModal
          eyebrow={event.name}
          photoCount={photos.length}
          eventPhotoCount={event.photoCount}
          onClose={() => setSearchOpen(false)}
          onSubmitBib={onSubmitBib}
        />
      )}

      {previewPhoto && previewIndex !== null && (
        <PhotoPreviewMount
          event={event}
          photo={previewPhoto}
          index={previewIndex}
          total={visible.length}
          onClose={() => setPreviewIndex(null)}
          onPrev={
            previewIndex > 0
              ? () => setPreviewIndex(previewIndex - 1)
              : undefined
          }
          onNext={
            previewIndex < visible.length - 1
              ? () => setPreviewIndex(previewIndex + 1)
              : undefined
          }
        />
      )}

      <Footer />
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
        thumbnailUrl: "",
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
      thumbnailUrl: "",
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
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik · Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
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
