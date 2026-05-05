"use client";

import {
  useState,
  useMemo,
  useEffect,
  useRef,
  type FormEvent,
  type RefObject,
} from "react";
import Link from "next/link";
import { useSearchParams, useRouter, usePathname } from "next/navigation";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import { cn } from "@/lib/utils";
import type { EventDetail } from "@/types/event";
import { type MockPhoto } from "./mock-photos";
import { PhotoPreviewCard } from "@/components/photos/photo-preview-card";

type Mode = "cockpit" | "browse";
type PanelMode = "bib" | "selfie";

interface Props {
  event: EventDetail;
  photos: MockPhoto[];
}

const TONE_COLORS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
];

export function EventCockpit({ event, photos }: Props) {
  const sp = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const initialBibParam = (sp.get("bib") ?? "").trim().toUpperCase();
  const initialBrowse = sp.get("browse") === "1";
  const initialMode: Mode =
    initialBrowse || initialBibParam ? "browse" : "cockpit";

  const [mode, setMode] = useState<Mode>(initialMode);
  const [panelMode, setPanelMode] = useState<PanelMode>("bib");
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
  panelMode: PanelMode;
  onPanelModeChange: (m: PanelMode) => void;
  onBrowseAll: () => void;
}) {
  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit(bibInput);
  };

  return (
    <>
      <div className="bg-bone">
        <div className="max-w-7xl mx-auto px-6 md:px-10 pt-5 md:pt-6">
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

function BibPanel({
  bibInput,
  onBibChange,
  onSubmit,
  onSwitchToSelfie,
  photoCount,
  eventPhotoCount,
  inputRef,
}: {
  bibInput: string;
  onBibChange: (v: string) => void;
  onSubmit: (e: FormEvent) => void;
  onSwitchToSelfie: () => void;
  photoCount: number;
  eventPhotoCount: number;
  inputRef?: RefObject<HTMLInputElement | null>;
}) {
  return (
    <form onSubmit={onSubmit} className="mt-8">
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mb-3">
        Your bib number
      </p>
      <label className="block">
        <span className="sr-only">Bib number</span>
        <input
          ref={inputRef}
          type="text"
          name="bib"
          inputMode="text"
          autoComplete="off"
          autoCapitalize="characters"
          value={bibInput}
          onChange={(e) => onBibChange(e.target.value)}
          placeholder="B-4082"
          className="block w-full border-b border-line bg-transparent focus:border-fresh outline-none font-mono tracking-[0.25em] text-lg py-3 placeholder:text-slate-soft text-ink"
        />
      </label>
      <button
        type="submit"
        className="mt-5 inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-xs transition-colors"
      >
        Search by bib →
      </button>

      <div className="mt-7 flex items-center gap-3">
        <span className="h-px flex-1 bg-line" />
        <span className="font-mono uppercase tracking-[0.3em] text-[9px] text-slate-soft">
          or
        </span>
        <span className="h-px flex-1 bg-line" />
      </div>

      <button
        type="button"
        onClick={onSwitchToSelfie}
        className="mt-7 inline-flex items-center border border-ink hover:bg-ink hover:text-bone text-ink px-6 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-xs transition-colors"
      >
        Upload a selfie →
      </button>

      <p className="mt-7 font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
        <span className="tnum">{photoCount.toLocaleString()}</span>{" "}
        {photoCount === 1 ? "photo" : "photos"}
        {eventPhotoCount > photoCount ? (
          <>
            {" "}
            of <span className="tnum">{eventPhotoCount.toLocaleString()}</span>
          </>
        ) : null}{" "}
        · free to search
      </p>
    </form>
  );
}

function SelfiePendingPanel({ onSwitchToBib }: { onSwitchToBib: () => void }) {
  return (
    <div className="mt-8" style={{ animation: "fade-in 0.3s ease-out both" }}>
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate mb-3">
        Selfie match
      </p>
      <div className="rounded-xl border border-line bg-bone-deep px-5 py-6">
        <p className="font-display text-xl font-medium text-ink leading-snug">
          Coming soon.
        </p>
        <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
          We&apos;ll match your face against every photo in this race once the
          AI service is wired in. For now, search by bib.
        </p>
      </div>
      <button
        type="button"
        onClick={onSwitchToBib}
        className="mt-6 inline-flex items-center font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-fresh transition-colors"
      >
        ← Use bib instead
      </button>
    </div>
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

/* ─────────────── BUY-ALL BAR (browse mode, when filtered) ─────────────── */

function BuyAllBar({
  event,
  photos,
  total,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  total: number;
}) {
  const addItem = useCartStore((s) => s.addItem);
  const items = useCartStore((s) => s.items);
  const [pressed, setPressed] = useState(false);

  const allInCart =
    photos.length > 0 &&
    photos.every((p) => items.some((i) => i.photoId === p.id));

  const handleBuyAll = () => {
    for (const p of photos) {
      addItem({
        photoId: p.id,
        eventId: event.id,
        thumbnailUrl: "",
        price: p.price,
        bib: p.bib,
        eventName: event.name,
        eventSlug: event.slug,
        tone: p.tone,
        time: p.time,
      });
    }
    setPressed(true);
    setTimeout(() => setPressed(false), 2400);
  };

  if (photos.length === 0) return null;

  return (
    <div className="fixed bottom-0 inset-x-0 px-4 md:px-10 py-3 md:py-4 bg-bone/95 backdrop-blur-md border-t border-line z-30">
      <div className="max-w-7xl mx-auto flex items-center justify-between gap-3 md:gap-4">
        <div className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate hidden sm:flex items-center gap-3">
          <span>
            <span className="tnum text-ink">{photos.length}</span>{" "}
            {photos.length === 1 ? "photo" : "photos"}
          </span>
          <span className="h-3 w-px bg-line" aria-hidden="true" />
          <span>
            Total{" "}
            <span className="tnum text-ink">₱{total.toLocaleString()}</span>
          </span>
        </div>
        <button
          type="button"
          onClick={handleBuyAll}
          disabled={pressed || allInCart}
          className={cn(
            "ml-auto inline-flex items-center bg-fresh hover:bg-fresh-deep text-bone px-5 md:px-7 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-xs transition-colors whitespace-nowrap",
            (pressed || allInCart) && "opacity-90 cursor-default",
          )}
        >
          {pressed || allInCart
            ? `Added · ${photos.length} in cart ✓`
            : photos.length === 1
              ? `Buy 1 · ₱${total.toLocaleString()} →`
              : `Buy all ${photos.length} · ₱${total.toLocaleString()} →`}
        </button>
      </div>
    </div>
  );
}

function EmptyResult({
  event,
  bib,
  onClear,
}: {
  event: EventDetail;
  bib: string;
  onClear: () => void;
}) {
  const { title, body, showNotify } = emptyResultCopy(event.status, bib);

  return (
    <div className="px-6 md:px-10 py-16 md:py-24">
      <div className="max-w-2xl mx-auto rounded-2xl bg-bone-deep border border-line p-8 md:p-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-4">
          BIB {bib} · {event.name}
        </p>
        <p className="font-display text-3xl md:text-4xl font-medium text-ink tracking-tight leading-tight">
          {title}
        </p>
        <p className="mt-4 font-sans text-base md:text-lg text-ink-soft max-w-md">
          {body}
        </p>
        {showNotify && <NotifyForm bib={bib} />}
        <button
          type="button"
          onClick={onClear}
          className="mt-6 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-fresh transition-colors"
        >
          Or skim the full gallery →
        </button>
      </div>
    </div>
  );
}

function emptyResultCopy(
  status: EventDetail["status"],
  bib: string,
): { title: string; body: string; showNotify: boolean } {
  switch (status) {
    case "DRAFT":
    case "ACTIVE":
      return {
        title: "Still uploading.",
        body: `Photographers are still working through this race. Drop your email and we'll ping you the moment ${bib} appears.`,
        showNotify: true,
      };
    case "COMPLETED":
      return {
        title: "Bib not found.",
        body: `All photos for this race have been uploaded — ${bib} isn't in there. Double-check the number, or skim the wall.`,
        showNotify: false,
      };
    case "ARCHIVED":
      return {
        title: "This race has wrapped.",
        body: `Photos for ${bib} never landed in this archive. The wall's still here if you want to skim.`,
        showNotify: false,
      };
    default: {
      const _exhaustive: never = status;
      return _exhaustive;
    }
  }
}

function NotifyForm({ bib }: { bib: string }) {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);
  if (submitted) {
    return (
      <p className="mt-8 font-mono uppercase tracking-[0.25em] text-[10px] text-fresh">
        ✓ We&apos;ll email you when bib {bib} appears.
      </p>
    );
  }
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (email.trim()) setSubmitted(true);
      }}
      className="mt-8 flex flex-col sm:flex-row items-stretch sm:items-end gap-3 max-w-md"
    >
      <label className="flex-1">
        <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate block mb-2">
          Notify me
        </span>
        <input
          type="email"
          required
          placeholder="you@email.com"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="block w-full border-b border-line bg-transparent focus:border-fresh outline-none font-sans text-base py-2 placeholder:text-slate-soft text-ink"
        />
      </label>
      <button
        type="submit"
        className="bg-ink hover:bg-ink-soft text-bone px-5 py-3 rounded-full font-mono uppercase tracking-[0.2em] text-xs transition-colors whitespace-nowrap"
      >
        Notify me →
      </button>
    </form>
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
        <div className="max-w-7xl mx-auto px-6 md:px-10 py-3 flex items-center justify-between gap-4">
          <button
            type="button"
            onClick={() => setSearchOpen(true)}
            aria-haspopup="dialog"
            className="inline-flex items-center gap-2.5 font-mono uppercase tracking-[0.25em] text-[10px] text-ink hover:text-fresh transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
          >
            <svg
              viewBox="0 0 16 16"
              className="size-3.5 text-slate"
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
            <span>
              {isFiltered ? `Search · ${bibFilter}` : "Find your photos"}
            </span>
          </button>
          {isFiltered ? (
            <button
              type="button"
              onClick={onClearBib}
              className="inline-flex items-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-slate hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
            >
              <span>Clear filter</span>
              <span aria-hidden="true">✕</span>
            </button>
          ) : (
            <span className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft hidden sm:inline">
              <span className="tnum text-ink">{photos.length}</span> photos
            </span>
          )}
        </div>
      </div>

      <div className="flex-1 flex flex-col">
        {isFiltered && visible.length === 0 ? (
          <EmptyResult event={event} bib={bibFilter} onClear={onClearBib} />
        ) : (
          <div className="px-6 md:px-10 py-10 md:py-14 pb-20">
            <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
              {visible.map((p, i) => (
                <PhotoTile
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
        <SearchModal
          event={event}
          photos={photos}
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

/* ─────────────── SEARCH MODAL (browse mode) ─────────────── */

function SearchModal({
  event,
  photos,
  onClose,
  onSubmitBib,
}: {
  event: EventDetail;
  photos: MockPhoto[];
  onClose: () => void;
  onSubmitBib: (b: string) => void;
}) {
  const [bibInput, setBibInput] = useState("");
  const [panelMode, setPanelMode] = useState<PanelMode>("bib");
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const bibInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    const previouslyFocused =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;

    const focusables = () =>
      dialogRef.current
        ? Array.from(
            dialogRef.current.querySelectorAll<HTMLElement>(
              'button:not([tabindex="-1"]):not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
            ),
          ).filter((el) => !el.hasAttribute("disabled"))
        : [];

    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
        return;
      }
      if (e.key === "Tab") {
        const items = focusables();
        if (items.length === 0) return;
        const first = items[0];
        const last = items[items.length - 1];
        const active = document.activeElement;
        if (e.shiftKey) {
          if (active === first || !dialogRef.current?.contains(active)) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last || !dialogRef.current?.contains(active)) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };

    document.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
      previouslyFocused?.focus();
    };
  }, [onClose]);

  useEffect(() => {
    if (panelMode === "bib") {
      bibInputRef.current?.focus();
    }
  }, [panelMode]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!bibInput.trim()) return;
    onClose();
    onSubmitBib(bibInput);
  };

  return (
    <div
      ref={dialogRef}
      role="dialog"
      aria-modal="true"
      aria-label="Find your photos"
      className="fixed inset-0 z-50 flex items-center justify-center px-4 py-6 md:p-10"
    >
      <button
        type="button"
        onClick={onClose}
        aria-label="Close search"
        tabIndex={-1}
        className="absolute inset-0 bg-ink/35 backdrop-blur-sm cursor-default"
        style={{ animation: "fade-in 0.2s ease-out both" }}
      />
      <div
        className="relative w-full max-w-md"
        style={{ animation: "fade-up 0.35s ease-out both" }}
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Close search"
          className="absolute -top-3 -right-3 z-10 size-9 rounded-full bg-ink text-bone flex items-center justify-center hover:bg-ink-soft transition-colors shadow-[0_8px_20px_-8px_rgba(17,17,17,0.45)] focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
        >
          <svg
            viewBox="0 0 16 16"
            className="size-3.5"
            fill="none"
            aria-hidden="true"
          >
            <path
              d="M3 3 L13 13 M13 3 L3 13"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
            />
          </svg>
        </button>
        <div className="rounded-2xl bg-bone border border-line shadow-[0_24px_60px_-20px_rgba(17,17,17,0.45)] p-8 md:p-10 max-h-[85vh] overflow-y-auto">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate mb-5">
            {event.name}
          </p>
          <h2 className="font-display text-4xl md:text-5xl font-medium tracking-tight leading-[0.95]">
            Find your
            <br />
            <span className="text-fresh">photos.</span>
          </h2>

          {panelMode === "bib" ? (
            <BibPanel
              bibInput={bibInput}
              onBibChange={setBibInput}
              onSubmit={handleSubmit}
              onSwitchToSelfie={() => setPanelMode("selfie")}
              photoCount={photos.length}
              eventPhotoCount={event.photoCount}
              inputRef={bibInputRef}
            />
          ) : (
            <SelfiePendingPanel
              onSwitchToBib={() => setPanelMode("bib")}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ─────────────── PHOTO TILE ─────────────── */

function PhotoTile({
  event,
  photo,
  index,
  onOpen,
}: {
  event: EventDetail;
  photo: MockPhoto;
  index: number;
  onOpen: () => void;
}) {
  const inCart = useCartStore((s) =>
    s.items.some((i) => i.photoId === photo.id),
  );
  const addItem = useCartStore((s) => s.addItem);
  const removeItem = useCartStore((s) => s.removeItem);
  const startExpressCheckout = useUiStore((s) => s.startExpressCheckout);
  const openCheckout = useUiStore((s) => s.openCheckout);

  const wide = photo.span === "wide";
  const colorIdx = photo.tone % TONE_COLORS.length;
  const opacity = 0.7 + (index % 3) * 0.1;
  const hasImage = Boolean(photo.imageUrl);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageFailed, setImageFailed] = useState(false);

  useEffect(() => {
    setImageLoaded(false);
    setImageFailed(false);
  }, [photo.id, photo.imageUrl]);

  const handleToggleCart = (e: React.MouseEvent) => {
    e.stopPropagation();
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

  const handleBuyNow = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inCart) {
      openCheckout();
      return;
    }
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

  const fadeRule =
    "opacity-100 md:opacity-60 md:group-hover:opacity-100";

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
        aria-label={
          photo.bib
            ? `Open preview of photo ${photo.bib}`
            : "Open preview of untagged crowd photo"
        }
        className="block w-full h-full text-left rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        <div
          className="relative overflow-hidden rounded-xl h-full transition-transform duration-300 group-hover:-translate-y-[2px]"
          style={{
            backgroundColor: TONE_COLORS[colorIdx],
            opacity,
          }}
        >
          {hasImage && !imageFailed && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={photo.imageUrl ?? ""}
              alt={
                photo.alt ??
                (photo.bib
                  ? `Race photo of bib ${photo.bib}`
                  : "Untagged race photo")
              }
              onLoad={() => setImageLoaded(true)}
              onError={() => setImageFailed(true)}
              className={cn(
                "absolute inset-0 w-full h-full object-cover transition-opacity duration-500",
                imageLoaded ? "opacity-100" : "opacity-0",
              )}
              draggable={false}
            />
          )}
          <span
            aria-hidden="true"
            className="absolute inset-0 flex items-center justify-center bg-ink/0 group-hover:bg-ink/30 transition-colors duration-300"
          >
            <span className="font-mono uppercase tracking-[0.3em] text-[10px] text-bone/0 group-hover:text-bone/95 transition-colors duration-300">
              View →
            </span>
          </span>
        </div>
      </button>
      <div className="absolute bottom-3 right-3 flex items-center gap-1.5">
        <button
          type="button"
          onClick={handleToggleCart}
          aria-pressed={inCart}
          aria-label={
            inCart
              ? `Remove ${photo.bib ?? "untagged photo"} from cart`
              : `Add ${photo.bib ?? "untagged photo"} to cart`
          }
          className={cn(
            "inline-flex items-center gap-1 px-2.5 py-1.5 rounded-full font-mono uppercase tracking-[0.2em] text-[9px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-all duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            inCart
              ? "bg-fresh text-bone hover:bg-fresh-deep"
              : cn(
                  "bg-bone/90 backdrop-blur-sm text-ink hover:bg-bone hover:scale-105",
                  fadeRule,
                ),
          )}
        >
          <span aria-hidden="true">{inCart ? "✓" : "+"}</span>
          <span>cart</span>
        </button>
        <button
          type="button"
          onClick={handleBuyNow}
          aria-label={`Buy ${photo.bib ?? "untagged photo"} now for ₱${photo.price}`}
          className={cn(
            "inline-flex items-center gap-1 px-2.5 py-1.5 rounded-full font-mono uppercase tracking-[0.2em] text-[9px] whitespace-nowrap",
            "shadow-[0_4px_12px_-2px_rgba(0,0,0,0.25)]",
            "transition-all duration-200",
            "focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
            "bg-bone/90 backdrop-blur-sm text-ink hover:bg-fresh hover:text-bone hover:scale-105",
            fadeRule,
          )}
        >
          <span>buy</span>
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </div>
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
