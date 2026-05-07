"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { ProfileShellFooter } from "@/components/profile-shell";
import { PhotoPreviewCard } from "@/components/photos/photo-preview-card";
import { PhotoMosaicTile } from "@/components/events/photo-mosaic-tile";
import { FindPhotosModal } from "@/components/events/find-photos-modal";
import { BuyAllBar } from "@/components/events/buy-all-bar";
import { BibEmptyResult } from "@/components/events/bib-empty-result";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { useUrlState } from "@/hooks/use-url-state";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import { isReservedHandle } from "@/lib/reserved-handles";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import {
  PHOTOGRAPHER_EVENTS,
  type EventState,
} from "@/lib/photographer-mock";
import {
  getPhotographerByHandle,
  getPhotographerCoverage,
  generatePhotographerPhotos,
  type PhotographerProfile,
} from "@/lib/photographer-registry";
import { MOCK_EVENT_DETAILS } from "@/app/events/mock-events";
import {
  usePhotographerSettingsStore,
  BRAND_COLOR_HEX,
} from "@/store/photographer-settings-store";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/app/events/[slug]/mock-photos";

// Per-photographer public gallery — what runners land on when they click a
// watermark URL on a photo. Filtered to only this photographer's photos for
// the named event, but otherwise mirrors `/events/[slug]?browse=1`:
// mosaic photo grid + per-tile +cart/buy buttons + sticky "Find your photos"
// search → FindPhotosModal + BuyAllBar when filtered + BibEmptyResult on
// no-match. The bib filter scope is photographer-only (their slice), not the
// full event pool.
//
// TODO(backend): swap registry resolve + photo generator for
// `api.get<{ profile, event, photos }>("/p/{handle}/events/{slug}")`.
export default function HandleEventPage() {
  const params = useParams<{ handle: string; slug: string }>();
  const rawHandle = Array.isArray(params.handle)
    ? params.handle[0]
    : params.handle;
  const rawSlug = Array.isArray(params.slug) ? params.slug[0] : params.slug;
  const handle = (rawHandle ?? "").trim().toLowerCase();
  const slug = (rawSlug ?? "").trim().toLowerCase();

  if (isReservedHandle(handle)) {
    notFound();
  }

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <PageBody handle={handle} slug={slug} />
      <ProfileShellFooter />
    </main>
  );
}

function PageBody({ handle, slug }: { handle: string; slug: string }) {
  const settings = usePhotographerSettingsStore();
  const event = MOCK_EVENT_DETAILS[slug];

  if (!event) {
    return <NotFoundBody handle={handle} reason="event" />;
  }

  const isOwnHandle = handle.length > 0 && settings.handle === handle;

  // Owner preview: build a synthetic profile + coverage from the live
  // settings store + PHOTOGRAPHER_EVENTS, so the photographer can preview
  // their own gallery directly from /dashboard/settings.
  if (isOwnHandle) {
    const ownerCoverage = PHOTOGRAPHER_EVENTS.find(
      (e) => e.slug === slug && e.photoCount > 0,
    );
    if (!ownerCoverage) {
      return <NotFoundBody handle={handle} reason="not-shot" event={event} />;
    }
    const ownerProfile: PhotographerProfile = {
      handle,
      displayName: settings.brandName.trim() || "Photographer",
      brandColor: settings.brandColor,
      bio: settings.bio,
      city: "Cebu",
      memberSince: "2024-08-15",
      cover: settings.cover
        ? { kind: "image", url: settings.cover.dataUrl }
        : null,
      watermarkLabel: "OWNER",
      events: [
        {
          eventSlug: ownerCoverage.slug,
          state: ownerCoverage.state,
          photoCount: ownerCoverage.photoCount,
          salesCount: ownerCoverage.salesCount,
        },
      ],
    };
    return <Gallery profile={ownerProfile} event={event} />;
  }

  const profile = getPhotographerByHandle(handle);
  if (!profile) {
    return <NotFoundBody handle={handle} reason="profile" />;
  }
  const coverage = getPhotographerCoverage(profile, slug);
  if (!coverage || coverage.photoCount === 0) {
    return <NotFoundBody handle={handle} reason="not-shot" event={event} />;
  }
  return <Gallery profile={profile} event={event} />;
}

/* ─────────────── GALLERY ─────────────── */

function Gallery({
  profile,
  event,
}: {
  profile: PhotographerProfile;
  event: EventDetail;
}) {
  const coverage = profile.events.find((e) => e.eventSlug === event.slug);
  const photoCount = coverage?.photoCount ?? 0;
  const accent =
    profile.brandColor !== "none"
      ? BRAND_COLOR_HEX[profile.brandColor]
      : null;

  const photos = useMemo(
    () => generatePhotographerPhotos(profile.handle, event, photoCount),
    [profile.handle, event, photoCount],
  );

  const [bibFilter, setBibFilter] = useUrlState<string>("bib", "", {
    parse: (raw) => raw.trim().toUpperCase(),
  });
  const [searchOpen, setSearchOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);
  const [loadedCount, setLoadedCount] = useState(PAGE_SIZE.PHOTO_INITIAL);

  const submitBib = (raw: string) => {
    const clean = raw.trim().toUpperCase();
    if (!clean) return;
    setBibFilter(clean);
  };

  const clearBib = () => {
    setBibFilter("");
  };

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
    setLoadedCount(PAGE_SIZE.PHOTO_INITIAL);
  }, [cleanedQuery]);

  const visibleSlice = useMemo(
    () => visible.slice(0, loadedCount),
    [visible, loadedCount],
  );

  useEffect(() => {
    if (previewIndex !== null && previewIndex >= visibleSlice.length) {
      setPreviewIndex(visibleSlice.length === 0 ? null : visibleSlice.length - 1);
    }
  }, [visibleSlice.length, previewIndex]);

  const total = visible.reduce((sum, p) => sum + p.price, 0);
  const showBuyAll = isFiltered && visible.length > 0;
  const previewPhoto =
    previewIndex !== null ? visibleSlice[previewIndex] ?? null : null;

  return (
    <div className="flex-1 flex flex-col">
      <header className="px-6 md:px-10 pt-8 md:pt-10 pb-8 md:pb-10 border-b border-line">
        <div className="max-w-7xl mx-auto">
          <Kicker
            as={Link}
            href={`/${profile.handle}`}
            className="group flex w-fit items-center gap-2 hover:text-ink transition-colors mb-7"
          >
            <span
              aria-hidden="true"
              className="transition-transform group-hover:-translate-x-0.5"
            >
              ←
            </span>
            <span>Back to {profile.displayName}</span>
          </Kicker>

          <PhotographerChip profile={profile} accent={accent} />

          <Kicker as="p" tnum className="mt-7 flex items-center gap-2 flex-wrap">
            <span>{formatLongDate(event.date, true)}</span>
            <span className="text-slate-soft">·</span>
            <CoverageStateChip state={coverage?.state ?? "open"} />
          </Kicker>
          <h1 className="font-display text-4xl md:text-6xl font-medium tracking-tight leading-[0.95] text-ink mt-4">
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
              <>{event.name}</>
            )}
          </h1>
          <p className="font-sans text-base md:text-lg text-ink-soft mt-4 max-w-xl leading-relaxed">
            {isFiltered ? (
              <>
                Photos from {profile.displayName} matching bib{" "}
                <span className="font-mono tnum">{bibFilter}</span>. Tap any to
                add to cart.
              </>
            ) : (
              <>
                <span className="tnum">{photoCount.toLocaleString()}</span>{" "}
                {photoCount === 1 ? "photo" : "photos"} from{" "}
                {profile.displayName} at {event.location}.
              </>
            )}
          </p>
          <Kicker as="p" tone="soft" className="mt-3">
            ₱<span className="tnum">{event.pricePerPhoto}</span> per photo ·
            free watermarked previews · pay once, download forever
          </Kicker>

          <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-3">
            <Kicker
              as={Link}
              href={`/events/${event.slug}`}
              className="hover:text-ink transition-colors px-3 py-1.5 rounded-full border border-line hover:border-slate"
            >
              See all photographers →
            </Kicker>
          </div>
        </div>
      </header>

      {photos.length > 0 && (
        <div className="sticky top-[3.75rem] z-20 bg-bone/90 backdrop-blur-md border-b border-line">
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
                <circle
                  cx="7"
                  cy="7"
                  r="4.5"
                  stroke="currentColor"
                  strokeWidth="1.5"
                />
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
            </Kicker>
            {isFiltered ? (
              <Kicker
                as="button"
                type="button"
                onClick={clearBib}
                className="shrink-0 inline-flex items-center gap-2 hover:text-ink transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
              >
                <span>Clear filter</span>
                <span aria-hidden="true">✕</span>
              </Kicker>
            ) : (
              <Kicker tone="soft" className="shrink-0 hidden sm:inline">
                <span className="tnum text-ink">{photos.length}</span> photos
              </Kicker>
            )}
          </div>
        </div>
      )}

      <div className="flex-1 px-6 md:px-10 py-10 md:py-14 pb-32">
        <div className="max-w-7xl mx-auto">
          {photos.length === 0 ? (
            <EmptyGalleryPanel displayName={profile.displayName} />
          ) : isFiltered && visible.length === 0 ? (
            <BibEmptyResult
              event={event}
              bib={bibFilter}
              onClear={clearBib}
              ctaLabel={`Or skim ${profile.displayName}'s gallery →`}
            />
          ) : (
            <>
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
                total={visible.length}
                increment={PAGE_SIZE.PHOTO_INCREMENT}
                onLoadMore={() =>
                  setLoadedCount((n) => n + PAGE_SIZE.PHOTO_INCREMENT)
                }
                countSuffix={isFiltered ? `· BIB ${bibFilter}` : undefined}
              />
            </>
          )}
        </div>
      </div>

      {showBuyAll && <BuyAllBar event={event} photos={visible} total={total} />}

      {searchOpen && (
        <FindPhotosModal
          eyebrow={`${profile.displayName} · ${event.name}`}
          photoCount={photos.length}
          eventPhotoCount={event.photoCount}
          onClose={() => setSearchOpen(false)}
          onSubmitBib={submitBib}
        />
      )}

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
    </div>
  );
}

/* ─────────────── PHOTOGRAPHER CHIP ─────────────── */

function PhotographerChip({
  profile,
  accent,
}: {
  profile: PhotographerProfile;
  accent: string | null;
}) {
  // Mobile: drops the URL tail (~230px wide with the mono uppercase 0.25em
  // tracking) which forced the display name to wrap mid-word inside the chip.
  // sm+ shows the full chip.
  return (
    <div className="inline-flex max-w-full items-center gap-3 rounded-full border border-line bg-bone-deep/30 px-3 py-2">
      <span
        aria-hidden="true"
        className="size-7 rounded-full flex items-center justify-center font-mono uppercase tracking-tight text-[9px] font-semibold text-bone shrink-0"
        style={{ backgroundColor: accent ?? "var(--ink)" }}
      >
        {getInitials(profile.displayName)}
      </span>
      <Kicker className="text-ink whitespace-nowrap">
        {profile.displayName}
      </Kicker>
      <span
        className="hidden sm:inline text-slate-soft text-[10px]"
        aria-hidden="true"
      >
        ·
      </span>
      <Kicker tone="soft" className="hidden sm:inline whitespace-nowrap">
        quickpitik.com/{profile.handle}
      </Kicker>
    </div>
  );
}

function CoverageStateChip({ state }: { state: EventState }) {
  const STATE_LABEL: Record<EventState, string> = {
    live: "LIVE",
    open: "OPEN",
    upcoming: "UPCOMING",
    past: "ARCHIVED",
  };
  if (state === "live") {
    return (
      <span className="inline-flex items-center gap-1.5">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh breathe"
        />
        <span className="text-ink">LIVE</span>
      </span>
    );
  }
  return <span>{STATE_LABEL[state]}</span>;
}

/* ─────────────── PREVIEW MOUNT (cart-aware) ─────────────── */

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

  const cartPayload = {
    photoId: photo.id,
    eventId: event.id,
    thumbnailUrl: "",
    price: photo.price,
    bib: photo.bib,
    eventName: event.name,
    eventSlug: event.slug,
    tone: photo.tone,
    time: photo.time,
  };

  const handleToggle = () => {
    if (inCart) {
      removeItem(photo.id);
    } else {
      addItem(cartPayload);
    }
  };

  const handleBuyNow = () => {
    onClose();
    if (inCart) {
      openCheckout();
      return;
    }
    startExpressCheckout();
    addItem(cartPayload);
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

/* ─────────────── EMPTY / NOT FOUND ─────────────── */

function EmptyGalleryPanel({ displayName }: { displayName: string }) {
  return (
    <div className="border border-dashed border-line rounded-2xl p-10 md:p-16 text-center max-w-2xl mx-auto">
      <Kicker as="p" tone="soft">
        Gallery
      </Kicker>
      <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink mt-4">
        Photos coming soon.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-md mx-auto">
        {displayName} hasn&apos;t uploaded for this event yet. Check back
        later.
      </p>
    </div>
  );
}

function NotFoundBody({
  handle,
  reason,
  event,
}: {
  handle: string;
  reason: "profile" | "event" | "not-shot";
  event?: EventDetail;
}) {
  const headline =
    reason === "profile"
      ? "No photographer here yet."
      : reason === "event"
        ? "Event not found."
        : `${event?.name ?? "This event"} isn't in their portfolio.`;
  const body =
    reason === "profile" ? (
      <>
        We couldn&apos;t find{" "}
        <span className="font-mono">quickpitik.com/{handle}</span>.
      </>
    ) : reason === "event" ? (
      <>
        The event slug doesn&apos;t match anything we&apos;re tracking.
      </>
    ) : (
      <>This photographer didn&apos;t cover this race.</>
    );

  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-20">
      <div className="text-center max-w-md">
        <Kicker as="p" tone="soft">
          Not found
        </Kicker>
        <h1 className="font-display text-4xl md:text-5xl font-medium tracking-tight text-ink mt-4">
          {headline}
        </h1>
        <p className="font-sans text-base text-ink-soft mt-3">{body}</p>
        <div className="mt-7 flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
          {handle && (reason === "event" || reason === "not-shot") && (
            <Link
              href={`/${handle}`}
              className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
            >
              ← View their portfolio
            </Link>
          )}
          <Link
            href={ROUTES.HOME}
            className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            Back to QuickPitik
          </Link>
        </div>
      </div>
    </div>
  );
}

/* ─────────────── HELPERS ─────────────── */

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
