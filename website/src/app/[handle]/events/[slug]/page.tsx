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
import { Skeleton } from "@/components/ui/skeleton";
import { useQuery } from "@tanstack/react-query";
import { useInfiniteList } from "@/hooks/use-infinite-list";
import { useUrlState } from "@/hooks/use-url-state";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { useCartStore } from "@/store/cart-store";
import { useUiStore } from "@/store/ui-store";
import { isReservedHandle } from "@/lib/reserved-handles";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { fetchEventDetail } from "@/lib/api-events";
import {
  fetchPublicPhotographer,
  fetchPublicPhotographerEventPhotos,
} from "@/lib/api-photographer-public";
import type { EventState } from "@/lib/photographer-mock";
import type { PhotographerProfile } from "@/lib/photographer-registry";
import { BRAND_COLOR_HEX } from "@/store/photographer-settings-store";
import type { EventDetail } from "@/types/event";
import type { MockPhoto } from "@/types/photo";

// Per-photographer public gallery — what runners land on when they click a
// watermark URL on a photo. Filtered to only this photographer's photos for
// the named event, but otherwise mirrors `/events/[slug]?browse=1`:
// mosaic photo grid + per-tile +cart/buy buttons + sticky "Find your photos"
// search → FindPhotosModal + BuyAllBar when filtered + BibEmptyResult on
// no-match. The bib filter scope is photographer-only (their slice), not the
// full event pool.
//
// Single data path: GET /public/photographers/{handle} + GET .../events/{slug}
// + GET .../events/{slug}/photos. No isOwner branch — owner-self sees the
// same shape as runners (watermarked thumbs, no edit affordances). The
// /profile page is the owner-only edit surface.
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
  // Profile + event detail fan out in parallel, both BE. Long stale times:
  // a public profile changes when the photographer edits settings (30 min),
  // an event's metadata changes on an admin edit (10 min) — and the presigned
  // cover URLs inside both live 1 h, which bounds how stale is safe.
  // useQuery (not the wrapper hook) so we can distinguish "still loading"
  // from "404 / not found". The wrapper collapses both into null which would
  // flash NotFoundBody before the BE response lands.
  const profileQuery = useQuery({
    queryKey: ["photographer", "public", handle],
    queryFn: () => fetchPublicPhotographer(handle),
    enabled: handle.length > 0,
    staleTime: 30 * 60_000,
  });
  const eventQuery = useQuery({
    queryKey: ["events", slug, "detail"],
    queryFn: () => fetchEventDetail(slug),
    enabled: slug.length > 0,
    staleTime: 10 * 60_000,
  });

  if (profileQuery.isLoading || eventQuery.isLoading) {
    return <GallerySkeleton />;
  }

  const event = eventQuery.data;
  const profile = profileQuery.data;

  if (!event) {
    return <NotFoundBody handle={handle} reason="event" />;
  }
  if (!profile) {
    return <NotFoundBody handle={handle} reason="profile" />;
  }

  const coverage = profile.events.find((e) => e.eventSlug === event.slug);
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

  // Real server pagination for the browse. bib filtering is client-side (the
  // public endpoint has no bib param — see "requires backend changes"), so
  // Load-more progresses the whole gallery and the bib filter is a view over
  // what's loaded; runners searching a specific bib use /events/[slug] (server
  // bib search) instead.
  const list = useInfiniteList<MockPhoto>({
    queryKey: ["photographer", "public", profile.handle, event.slug, "photos"],
    fetchPage: (offset, limit) =>
      fetchPublicPhotographerEventPhotos(profile.handle, event.slug, {
        offset,
        limit,
      }),
    limit: PAGE_SIZE.PHOTO_INCREMENT,
    staleTime: 60_000,
  });
  const photos = list.items;

  const [bibFilter, setBibFilter] = useUrlState<string>("bib", "", {
    parse: (raw) => raw.trim().toUpperCase(),
  });
  const [searchOpen, setSearchOpen] = useState(false);
  const [previewIndex, setPreviewIndex] = useState<number | null>(null);

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
    if (previewIndex !== null && previewIndex >= visible.length) {
      setPreviewIndex(visible.length === 0 ? null : visible.length - 1);
    }
  }, [visible.length, previewIndex]);

  const total = visible.reduce((sum, p) => sum + p.price, 0);
  const showBuyAll = isFiltered && visible.length > 0;
  const previewPhoto =
    previewIndex !== null ? visible[previewIndex] ?? null : null;

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
          <h1 className="font-display text-4xl md:text-6xl font-extrabold tracking-tight leading-[0.95] text-ink mt-4">
            {isFiltered ? (
              visible.length === 0 ? (
                // The bib filter is a client view over loaded pages — while
                // unsearched pages remain (or a fetch is in flight), the
                // negative isn't known yet.
                list.hasNextPage || list.isFetching ? (
                  "Searching…"
                ) : (
                  "No matches yet."
                )
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
        <div className="sticky top-[var(--site-header-h)] z-20 bg-bone/90 backdrop-blur-md border-b border-line">
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
                <span className="tnum text-ink">{list.total}</span> photos
              </Kicker>
            )}
          </div>
        </div>
      )}

      <div className="flex-1 px-6 md:px-10 py-10 md:py-14 pb-32">
        <div className="max-w-7xl mx-auto">
          {list.isLoading ? (
            // First photos page is a client fetch with no SSR seed — don't
            // show "Photos coming soon" (or an error's empty fallback) while
            // it's still in flight.
            <div
              className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]"
              aria-busy="true"
            >
              {Array.from({ length: 8 }, (_, i) => (
                <Skeleton key={i} className="h-full w-full rounded-xl" />
              ))}
            </div>
          ) : list.error && photos.length === 0 ? (
            <div className="border border-dashed border-line rounded-2xl p-10 md:p-16 text-center max-w-2xl mx-auto">
              <Kicker as="p" tone="soft">
                Gallery
              </Kicker>
              <p className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink mt-4">
                Couldn&apos;t load photos.
              </p>
              <Kicker
                as="button"
                type="button"
                onClick={list.refetch}
                className="mt-6 inline-flex text-ink underline decoration-line underline-offset-4 hover:decoration-ink"
              >
                Try again ↻
              </Kicker>
            </div>
          ) : photos.length === 0 ? (
            <EmptyGalleryPanel displayName={profile.displayName} />
          ) : isFiltered && visible.length === 0 && !list.hasNextPage ? (
            <BibEmptyResult
              event={event}
              bib={bibFilter}
              onClear={clearBib}
              ctaLabel={`Or skim ${profile.displayName}'s gallery →`}
            />
          ) : (
            <>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
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
              {/* Load-more progresses the whole gallery; when a bib is active
                  the grid above is the filtered view over what's loaded. */}
              <LoadMoreButton
                shown={photos.length}
                total={list.total}
                increment={PAGE_SIZE.PHOTO_INCREMENT}
                onLoadMore={list.fetchNextPage}
                isLoading={list.isFetchingNextPage}
                countSuffix={isFiltered ? `· BIB ${bibFilter}` : undefined}
              />
            </>
          )}
        </div>
      </div>

      {showBuyAll && <BuyAllBar event={event} photos={visible} total={total} />}

      {searchOpen && (
        <FindPhotosModal
          eventSlug={event.slug}
          eyebrow={`${profile.displayName} · ${event.name}`}
          photoCount={list.total}
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
    thumbnailUrl: photo.imageUrl ?? "",
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
      eventDate={event.date}
      index={index + 1}
      total={total}
      // Every photo on this page is this photographer's, and their name is in
      // the header — crediting each shot back to the page you're standing on
      // is noise.
      showPhotographerCredit={false}
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

/* ─────────────── EMPTY / NOT FOUND / SKELETON ─────────────── */

function GallerySkeleton() {
  return (
    <div className="flex-1">
      <div className="px-6 md:px-10 pt-8 md:pt-10 pb-8 md:pb-10 border-b border-line">
        <div className="max-w-7xl mx-auto space-y-5">
          <Skeleton className="h-4 w-32" />
          <Skeleton className="h-9 w-64 rounded-full" />
          <Skeleton className="h-12 md:h-14 w-3/4" />
          <Skeleton className="h-5 w-1/2" />
        </div>
      </div>
      <div className="px-6 md:px-10 py-10">
        <div className="max-w-7xl mx-auto grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
          {[0, 1, 2, 3, 4, 5, 6, 7].map((i) => (
            <Skeleton key={i} className="aspect-square" />
          ))}
        </div>
      </div>
    </div>
  );
}

function EmptyGalleryPanel({ displayName }: { displayName: string }) {
  return (
    <div className="border border-dashed border-line rounded-2xl p-10 md:p-16 text-center max-w-2xl mx-auto">
      <Kicker as="p" tone="soft">
        Gallery
      </Kicker>
      <p className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink mt-4">
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
        <h1 className="font-display text-4xl md:text-5xl font-extrabold tracking-tight text-ink mt-4">
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
