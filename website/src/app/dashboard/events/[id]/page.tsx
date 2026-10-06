"use client";

import Link from "next/link";
import { notFound, useParams, useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { EventState, ListEvent } from "@/types/event";
import { EventCouponModal } from "@/components/dashboard/event-coupon-modal";
import { PhotographerPurchasePreview, type PurchaseSelection } from "@/components/dashboard/photographer-purchase-preview";
import { MyEventForm } from "@/components/dashboard/my-event-form";
import { SiteHeader } from "@/components/layout/site-header";
import {
  PhotoPreviewCard,
  type PhotoPreviewItem,
} from "@/components/photos/photo-preview-card";
import {
  BTN_DANGER,
  BTN_GHOST,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { LoadMoreButton } from "@/components/ui/load-more-button";
import { Modal } from "@/components/ui/modal";
import { Skeleton, TileSkeleton } from "@/components/ui/skeleton";
import { useConfirmation } from "@/hooks/use-confirmation";
import {
  usePhotographerEventDetail,
  usePhotographerEventMetadataVisibility,
  usePhotographerEventPhotos,
} from "@/hooks/use-photographer-data";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { usePhotographerPurchasedPhotos } from "@/hooks/use-photographer-purchases";
import { useToast } from "@/hooks/use-toast";
import {
  fetchPhotographerPhotoDownload,
  permanentDeletePhotographerPhotos,
  preparePhotographerBundle,
  restorePhotographerPhotos,
  trashPhotographerPhotos,
  updateMyEvent,
  putEventMetadataVisibility,
  type EventMetadataVisibilityDto,
  type PhotographerEventDetail,
} from "@/lib/api-photographer";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { describePricing, isOwnedEventLive } from "@/lib/photographer-events";
import type {
  PhotographerEventSummary,
  PhotographerLibraryPhoto,
} from "@/lib/photographer-mock";
import { PAGE_SIZE } from "@/lib/pagination-config";
import { isInAppBrowser } from "@/lib/in-app-browser";
import { savePhoto, showInAppSaveHint } from "@/lib/save-photo";
import { cn, copyToClipboard, triggerDownload } from "@/lib/utils";

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

  const { detail: liveDetail, isMissing, error, refetch } = usePhotographerEventDetail(id ?? null, { includeRemoved: true });
  // Owned-event controls (V46): edit modal + withdraw of a parked pricing
  // change. Hooks stay above the early returns.
  const [editOpen, setEditOpen] = useState(false);
  const [couponOpen, setCouponOpen] = useState(false);
  const queryClient = useQueryClient();
  const { showToast } = useToast();

  // Only events the photographer actually covers render this page. A 404 means
  // no event_photographer row, or the event is gone — either way the page
  // doesn't exist for them.
  if (isMissing) {
    notFound();
  }

  if (error) {
    return <main className="bg-bone text-ink min-h-screen flex flex-col"><SiteHeader /><div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 py-12"><p role="alert" className="font-sans text-base">Couldn&apos;t load this event.</p><button type="button" onClick={refetch} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-4")}>Retry</button></div></main>;
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

  if (liveDetail.deletedAt) {
    return <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <BackChip />
        <Hero event={event} />
        <p className="mb-8 font-mono text-sm uppercase tracking-wide text-slate">Removed event</p>
        <Stats photographer={photographer} />
        <section>
          <div role="tablist" aria-label="Photo library" className="mb-6 border-b border-line pb-4"><TabPill active onClick={() => { }}>Purchased</TabPill></div>
          <PurchasedGallery eventId={liveDetail.id} />
        </section>
      </div>
    </main>;
  }

  const refreshEvents = () =>
    queryClient.invalidateQueries({ queryKey: ["photographer", "events"] });

  async function handleWithdraw() {
    if (!liveDetail) return;
    try {
      await updateMyEvent(liveDetail.id, { withdrawPendingChange: true });
      await refreshEvents();
      showToast({ kind: "success", message: "Pricing request withdrawn." });
    } catch {
      showToast({ kind: "error", message: "Couldn't withdraw the request." });
    }
  }

  // Owned event (V46): the review band and the edit modal ride on both
  // layouts below. Uploads (and the share band) only make sense once live.
  const owned = liveDetail.ownedByMe === true;
  const uploadsOpen = !owned || isOwnedEventLive(liveDetail);
  // Coupons ride on coverage, not ownership (V49): any paid event this
  // photographer covers — admin-created included — can carry their code. Free
  // events have nothing to discount. Owned events get the button inside
  // OwnedBand; covered admin events get a slim band of their own.
  const paid = liveDetail.pricingMode !== "free";
  const couponUi = paid ? (
    <>
      {!owned && <CoverageBand onCoupon={() => setCouponOpen(true)} />}
      <EventCouponModal
        isOpen={couponOpen}
        onClose={() => setCouponOpen(false)}
        events={[liveDetail]}
        initialEventId={liveDetail.id}
        lockEvent
      />
    </>
  ) : null;
  const ownedUi = owned ? (
    <>
      <OwnedBand
        detail={liveDetail}
        onEdit={() => setEditOpen(true)}
        onCoupon={paid ? () => setCouponOpen(true) : undefined}
        onWithdraw={handleWithdraw}
      />
      <Modal
        isOpen={editOpen}
        onClose={() => setEditOpen(false)}
        title="Edit event"
      >
        <MyEventForm
          event={liveDetail}
          onDone={(saved) => {
            void refreshEvents();
            setEditOpen(false);
            showToast({
              kind: "success",
              message:
                saved.reviewStatus === "change_pending"
                  ? "Saved. The pricing change waits for admin approval."
                  : saved.reviewStatus === "pending"
                    ? "Saved and resubmitted for review."
                    : "Event updated.",
            });
          }}
          onCancel={() => setEditOpen(false)}
        />
      </Modal>
    </>
  ) : null;

  // One page for every state. An owned event that is pending/rejected keeps
  // uploads and the gallery closed (the OwnedBand explains why). Once uploads
  // are open the library carries BOTH the Active gallery and the Deleted
  // Photos tab — so a photographer who trashed everything can still reach and
  // restore their photos. ShareHeroBand is held back until there is at least
  // one active photo: a public gallery URL with nothing behind it dead-ends
  // the photographer's followers.
  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <div className="flex-1 max-w-7xl w-full mx-auto px-6 md:px-10 pt-8 md:pt-12 pb-16 md:pb-24">
        <BackChip />
        <Hero event={event} />
        {ownedUi}
        {couponUi}
        <EventMetadataVisibility event={liveDetail} />
        {uploadsOpen && (
          <>
            {liveDetail.photoCount > 0 && (
              <ShareHeroBand event={event} handle={liveDetail.handle ?? null} />
            )}
            <Stats photographer={photographer} />
            <PhotoLibrary event={event} eventId={liveDetail.id} />
          </>
        )}
      </div>
    </main>
  );
}

function EventMetadataVisibility({ event }: { event: PhotographerEventDetail }) {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const visibility = usePhotographerEventMetadataVisibility(event.id);
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const [draft, setDraft] = useState<boolean | null>(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const serverValue = visibility.data?.metadataPublic;
  const sessionOwner = session.key[0];
  const sessionRole = session.key[1];
  const sessionVersion = session.key[2];

  useEffect(() => {
    setDraft(null);
    setDirty(false);
    setSaving(false);
  }, [event.id, sessionOwner, sessionRole, sessionVersion]);

  useEffect(() => {
    if (!dirty && visibility.isSuccess) setDraft(serverValue ?? null);
  }, [event.id, sessionOwner, sessionRole, sessionVersion, dirty, visibility.isSuccess, serverValue]);

  async function save() {
    if (!session.enabled || !visibility.isSuccess || draft === null || draft === visibility.data.metadataPublic || saving) return;
    const isCurrent = captureAuthSession();
    setSaving(true);
    try {
      const saved = await putEventMetadataVisibility(event.id, { metadataPublic: draft });
      if (!isCurrent()) return;
      const visibilityKey = ["photographer", "events", event.id, "metadata-visibility", ...session.key];
      await queryClient.cancelQueries({ queryKey: visibilityKey, exact: true });
      if (!isCurrent()) return;
      queryClient.setQueryData<EventMetadataVisibilityDto>(
        visibilityKey, saved,
      );
      void queryClient.invalidateQueries({ queryKey: ["events", event.slug, "photos"] });
      if (event.handle) void queryClient.invalidateQueries({ queryKey: ["photographer", "public", event.handle, event.slug, "photos"] });
      // M7's public photo metadata queries use this prefix; no M7 hook exists yet.
      queryClient.removeQueries({ queryKey: ["photo", "metadata", "public"] });
      setDraft(saved.metadataPublic);
      setDirty(false);
      showToast({ kind: "success", message: "Metadata visibility saved." });
    } catch {
      if (isCurrent()) showToast({ kind: "error", message: "Couldn't save metadata visibility." });
    } finally {
      if (isCurrent()) setSaving(false);
    }
  }

  if (!session.enabled) return null;
  return (
    <section className="mb-10 md:mb-12 rounded-2xl border border-line bg-bone-deep/30 p-6 md:p-8">
      <Kicker as="p" tone="soft">Photo metadata</Kicker>
      {visibility.isError ? (
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <p className="font-sans text-sm text-slate">Couldn&apos;t load metadata visibility.</p>
          <button type="button" onClick={() => void visibility.refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry metadata visibility</button>
        </div>
      ) : !visibility.isSuccess || draft === null ? (
        <p className="mt-3 font-sans text-sm text-slate" role="status">Loading metadata visibility…</p>
      ) : (
        <>
          <label className="mt-3 flex min-h-[44px] items-center gap-3 font-sans text-base font-semibold text-ink">
            <input
              type="checkbox"
              checked={draft}
              onChange={(change) => {
                setDraft(change.target.checked);
                setDirty(change.target.checked !== visibility.data.metadataPublic);
              }}
              disabled={saving}
              aria-describedby="event-metadata-help event-metadata-boundary"
              className="size-5 shrink-0 accent-fresh disabled:cursor-not-allowed"
            />
            Show metadata publicly for my photos
          </label>
          <p id="event-metadata-help" className="mt-3 font-sans text-sm text-slate">Only your photos in this event. You can always view your own metadata.</p>
          <p id="event-metadata-boundary" className="mt-2 font-sans text-sm text-slate">This controls the metadata popup. Original downloads keep their embedded metadata.</p>
          <div className="mt-5 flex flex-wrap gap-2">
            <button type="button" onClick={() => void save()} disabled={saving || draft === visibility.data.metadataPublic} className={cn(BTN_PRIMARY, BTN_SIZE.sm)}>Save metadata visibility</button>
            <button type="button" onClick={() => { setDraft(visibility.data.metadataPublic); setDirty(false); }} disabled={saving || draft === visibility.data.metadataPublic} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Cancel metadata change</button>
          </div>
        </>
      )}
    </section>
  );
}

// Review band for a photographer-owned event (V46). States mirror the BE
// review lifecycle: pending (draft, invisible), rejected (+ admin note),
// live, or live with a pricing change parked for approval. The pricing trio
// on a live event is never edited in place — the band says what is live and
// what is requested, and the form behind "Edit event" files a request.
function OwnedBand({
  detail,
  onEdit,
  onCoupon,
  onWithdraw,
}: {
  detail: PhotographerEventDetail;
  onEdit: () => void;
  onCoupon?: () => void;
  onWithdraw: () => void;
}) {
  const status = detail.reviewStatus ?? "approved";
  const current = describePricing({
    pricingMode: detail.pricingMode ?? "paid",
    pricePerPhoto: detail.pricePerPhoto,
    watermarkPolicy: detail.watermarkPolicy ?? "platform",
  });
  const where = detail.visibility === "unlisted" ? "Unlisted" : "Public";

  let headline: string;
  let body: string;
  switch (status) {
    case "pending":
      headline = "Waiting for admin review.";
      body = `${where} · ${current}. Uploads open the moment an admin approves it.`;
      break;
    case "rejected":
      headline = "Sent back by admin.";
      body = detail.reviewNote
        ? `Reason: ${detail.reviewNote}`
        : "Edit the event and it goes back into review.";
      break;
    case "change_pending":
      headline = "Live · pricing change waiting for approval.";
      body = detail.pendingChange
        ? `Now ${current} → requested ${describePricing(detail.pendingChange)}. The gallery keeps the current pricing until an admin approves.`
        : current;
      break;
    default:
      headline = `Live · ${where}.`;
      body = detail.reviewNote
        ? `${current}. Last admin note: ${detail.reviewNote}`
        : current;
  }

  return (
    <section className="mb-10 md:mb-12 border border-line rounded-2xl bg-bone-deep/30 p-6 md:p-8 flex flex-col md:flex-row md:items-center gap-4 md:gap-6">
      <div className="flex-1 min-w-0">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Your event
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          {headline}
        </p>
        <p className="font-sans text-sm text-ink-soft mt-2 tnum">{body}</p>
      </div>
      <div className="flex flex-wrap gap-2 shrink-0">
        <button
          type="button"
          onClick={onEdit}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Edit event
        </button>
        {onCoupon && (
          <button
            type="button"
            onClick={onCoupon}
            className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
          >
            Event coupon
          </button>
        )}
        {status === "change_pending" && (
          <button
            type="button"
            onClick={onWithdraw}
            className={cn(BTN_GHOST, BTN_SIZE.sm)}
          >
            Withdraw request
          </button>
        )}
      </div>
    </section>
  );
}

// Coverage band for an event someone else runs (admin or another
// photographer). Same slab recipe as OwnedBand so the two read as siblings;
// its only action is the coupon, since editing belongs to the event's owner.
function CoverageBand({ onCoupon }: { onCoupon: () => void }) {
  return (
    <section className="mb-10 md:mb-12 border border-line rounded-2xl bg-bone-deep/30 p-6 md:p-8 flex flex-col md:flex-row md:items-center gap-4 md:gap-6">
      <div className="flex-1 min-w-0">
        <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
          Your coverage
        </p>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          Offer runners a code on your photos.
        </p>
        <p className="font-sans text-sm text-ink-soft mt-2">
          The discount comes out of your share and covers every paid photo you
          upload to this race.
        </p>
      </div>
      <div className="flex flex-wrap gap-2 shrink-0">
        <button
          type="button"
          onClick={onCoupon}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Event coupon
        </button>
      </div>
    </section>
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

function ShareHeroBand({
  event,
  handle,
}: {
  event: ListEvent;
  handle: string | null;
}) {
  const { showToast } = useToast();
  const [copied, setCopied] = useState(false);

  // The public gallery lives at quickpitik.com/{handle}/events/{slug}, and the
  // handle now comes from the backend event detail — not a client store that
  // could still hold the "your-handle" placeholder. No handle yet ⇒ there is no
  // shareable URL, so the actions stay disabled instead of copying a broken
  // link that dead-ends the photographer's followers.
  const hasHandle = handle !== null && handle.trim().length > 0;
  const path = hasHandle ? `/${handle}/events/${event.slug}` : "";
  const fullUrl =
    typeof window !== "undefined"
      ? `${window.location.origin}${path}`
      : `https://quickpitik.com${path}`;
  const displayUrl = hasHandle
    ? `quickpitik.com${path}`
    : "Set a public handle in Settings to share this gallery";

  async function handleCopy() {
    if (!hasHandle) return;
    const ok = await copyToClipboard(fullUrl);
    if (ok) {
      setCopied(true);
      showToast({ kind: "success", message: "Link copied." });
      setTimeout(() => setCopied(false), 2500);
    } else {
      showToast({
        kind: "error",
        message: "Couldn't copy. Select the link above and copy it manually.",
      });
    }
  }

  function shareTo(provider: "facebook" | "x" | "threads") {
    if (!hasHandle) return;
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

  async function handleInstagram() {
    if (!hasHandle) return;
    const ok = await copyToClipboard(fullUrl);
    showToast(
      ok
        ? {
            kind: "info",
            message: "Link copied. Paste into your IG bio or story.",
          }
        : {
            kind: "error",
            message:
              "Couldn't copy. Select the link above and copy it manually.",
          },
    );
  }

  function handleNativeShare() {
    if (!hasHandle) return;
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
      void handleCopy();
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
          disabled={!hasHandle}
          aria-label="Copy public gallery link"
          className={cn(
            "font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors inline-flex items-center justify-center gap-2 shrink-0",
            !hasHandle && "opacity-50 cursor-not-allowed hover:bg-transparent hover:text-ink",
          )}
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
            disabled={!hasHandle}
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
            disabled={!hasHandle}
          />
          <ShareChip
            onClick={() => shareTo("x")}
            dotColor="#000000"
            label="X"
            disabled={!hasHandle}
          />
          <ShareChip
            onClick={() => shareTo("threads")}
            dotColor="#444444"
            label="Threads"
            disabled={!hasHandle}
          />
          <ShareChip
            onClick={handleNativeShare}
            dotColor="var(--slate-soft)"
            label="More"
            disabled={!hasHandle}
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
  disabled,
}: {
  onClick: () => void;
  dotColor: string;
  label: string;
  /** Trailing qualifier for chips that don't open a share window. */
  hint?: string;
  ariaLabel?: string;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel ?? `Share to ${label}`}
      className={cn(
        "inline-flex items-center gap-2 font-sans text-sm font-medium border border-line bg-bone hover:bg-bone-deep hover:border-ink py-2 px-4 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
        disabled &&
          "opacity-50 cursor-not-allowed hover:bg-bone hover:border-line",
      )}
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

// Active + Deleted photo library. Tabs switch between the live gallery and the
// per-event trash. "Select" enters a touch-friendly selection mode (no hover
// affordances, so it works on a phone) whose bulk bar trashes / restores /
// permanently deletes; "Download all" streams a ZIP of active originals. Tap a
// tile (outside select mode) to open the owned-mode lightbox + single download.
function PhotoLibrary({
  event,
  eventId,
}: {
  event: ListEvent;
  eventId: string;
}) {
  const { showToast } = useToast();
  const { confirm } = useConfirmation();
  const queryClient = useQueryClient();
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedTab = searchParams.get("tab");

  const [section, setSection] = useState<"active" | "deleted" | "purchased">(
    requestedTab === "deleted" || requestedTab === "purchased" ? requestedTab : "active",
  );
  const [selecting, setSelecting] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  useEffect(() => {
    setSection(requestedTab === "deleted" || requestedTab === "purchased" ? requestedTab : "active");
  }, [requestedTab]);

  const active = usePhotographerEventPhotos(eventId, { deleted: false, enabled: section === "active" });
  // Only fetch the trash when its tab is open — most visits never touch it.
  const trash = usePhotographerEventPhotos(eventId, {
    deleted: true,
    enabled: section === "deleted",
  });
  const list = section === "active" ? active : trash;
  const photos = list.items;

  // Broad but correct: refreshes the detail (photoCount), both photo lists,
  // and the events list card — all keyed under ["photographer","events"].
  const invalidate = () =>
    queryClient.invalidateQueries({ queryKey: ["photographer", "events"] });

  function switchSection(next: "active" | "deleted" | "purchased") {
    if (next === section) return;
    setSection(next);
    router.replace(`${ROUTES.DASHBOARD_EVENTS}/${eventId}?tab=${next}`, { scroll: false });
    setSelecting(false);
    setSelected(new Set());
    setOpenIndex(null);
  }

  function exitSelect() {
    setSelecting(false);
    setSelected(new Set());
  }

  const toggle = useCallback((photoId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) next.delete(photoId);
      else next.add(photoId);
      return next;
    });
  }, []);

  const toggleAll = () =>
    setSelected((prev) =>
      prev.size === photos.length
        ? new Set()
        : new Set(photos.map((p) => p.id)),
    );

  async function runTrashOrRestore(kind: "trash" | "restore") {
    const ids = photos.filter((photo) => selected.has(photo.id) &&
      (kind === "trash" || !photo.deletionStartedAt)).map((photo) => photo.id);
    if (ids.length === 0) return;
    setBusy(true);
    try {
      const result = kind === "trash"
        ? await trashPhotographerPhotos(eventId, ids)
        : await restorePhotographerPhotos(eventId, ids);
      await invalidate();
      if (result.failed.length) setSelected(new Set(result.failed));
      else exitSelect();
      const n = result.affected;
      const remaining = ids.length - n;
      showToast({
        kind: remaining > 0 ? "info" : "success",
        message: (kind === "trash"
            ? `${n} photo${n > 1 ? "s" : ""} moved to Deleted.`
            : `${n} photo${n > 1 ? "s" : ""} restored.`) +
          (remaining > 0 ? ` ${remaining} unchanged. Refresh or retry the remaining photos.` : ""),
      });
    } catch {
      showToast({ kind: "error", message: "Something went wrong. Try again." });
    } finally {
      setBusy(false);
    }
  }

  async function runPermanent() {
    const ids = [...selected];
    if (ids.length === 0) return;
    const n = ids.length;
    const ok = await confirm({
      title: `Permanently delete ${n} photo${n > 1 ? "s" : ""}?`,
      message:
        "This can't be undone — the original files are removed from storage. Photos that have already sold are kept so their buyers can still download them.",
      confirmLabel: "Delete forever",
      danger: true,
    });
    if (!ok) return;
    setBusy(true);
    try {
      const res = await permanentDeletePhotographerPhotos(eventId, ids);
      await invalidate();
      if (res.failed?.length) setSelected(new Set(res.failed));
      else exitSelect();
      const del = res.deleted.length;
      const blocked = res.blockedSold.length;
      const pending = res.blockedPending?.length ?? 0;
      const failed = res.failed?.length ?? 0;
      showToast({
        kind: failed ? "error" : blocked || pending ? "info" : "success",
        message: `${del} permanently deleted.` +
          (blocked ? ` ${blocked} kept for buyers.` : "") +
          (pending ? ` ${pending} kept while payment is pending or being checked.` : "") +
          (failed ? ` ${failed} couldn't be deleted. Retry Delete forever to finish cleanup.` : ""),
      });
    } catch {
      showToast({ kind: "error", message: "Couldn't delete. Try again." });
    } finally {
      setBusy(false);
    }
  }

  async function handleDownload(photoIds?: string[]) {
    if (photoIds?.length === 0 || downloading) return;
    if (photoIds?.length === 1) {
      await handleDownloadSingle(photoIds[0]);
      return;
    }
    if (isInAppBrowser()) {
      showInAppSaveHint();
      return;
    }
    setDownloading(true);
    try {
      const { url } = await preparePhotographerBundle(eventId);
      // Top-level navigation to the tokened ZIP; the server sends
      // Content-Disposition: attachment, so the browser downloads without
      // leaving the page (same shape as the order-bundle download).
      triggerDownload(url, photoIds ? { photoIds: photoIds.join(",") } : undefined);
      showToast({ kind: "success", message: "Preparing your ZIP…" });
    } catch {
      showToast({
        kind: "error",
        message: "Couldn't start the download. Try again.",
      });
    } finally {
      setDownloading(false);
    }
  }

  // Lightbox (view + single-photo download), disabled while selecting.
  const handleClose = useCallback(() => setOpenIndex(null), []);
  const handlePrev = useCallback(
    () => setOpenIndex((i) => (i === null || i === 0 ? i : i - 1)),
    [],
  );
  const handleNext = useCallback(
    () =>
      setOpenIndex((i) => (i === null || i >= photos.length - 1 ? i : i + 1)),
    [photos.length],
  );
  const handleDownloadSingle = useCallback(async (photoId: string) => {
    setDownloading(true);
    try {
      const result = await fetchPhotographerPhotoDownload(photoId);
      if (result?.url) {
        const saved = await savePhoto(result.url);
        if (saved === "downloaded") showToast({ kind: "success", message: "Download started." });
        return;
      }
      showToast({ kind: "error", message: "Download unavailable — try again." });
    } catch (err) {
      console.error("[photographer/download] failed", err);
      showToast({ kind: "error", message: "Download failed. Try again." });
    } finally {
      setDownloading(false);
    }
  }, [showToast]);

  if (section === "purchased") {
    return <section>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-4 mb-6">
        <div role="tablist" aria-label="Photo library" className="flex flex-wrap gap-2">
          <TabPill active={false} onClick={() => switchSection("active")}>Active</TabPill>
          <TabPill active={false} onClick={() => switchSection("deleted")}>Deleted</TabPill>
          <TabPill active onClick={() => switchSection("purchased")}>Purchased</TabPill>
        </div>
      </div>
      <PurchasedGallery eventId={eventId} />
    </section>;
  }

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

  const gridClass =
    "grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]";

  return (
    <section>
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pb-4 mb-6">
        <div role="tablist" aria-label="Photo library" className="flex flex-wrap gap-2">
          <TabPill
            active={section === "active"}
            onClick={() => switchSection("active")}
          >
            Active
          </TabPill>
          <TabPill
            active={section === "deleted"}
            onClick={() => switchSection("deleted")}
          >
            Deleted
          </TabPill>
          <TabPill active={false} onClick={() => switchSection("purchased")}>Purchased</TabPill>
        </div>
        {!selecting && (
          <div className="flex items-center gap-2">
            {section === "active" && list.total > 0 && (
              <button
                type="button"
                onClick={() => handleDownload()}
                disabled={downloading}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                {downloading ? "Preparing…" : "Download all"}
              </button>
            )}
            {photos.length > 0 && (
              <button
                type="button"
                onClick={() => setSelecting(true)}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                Select
              </button>
            )}
          </div>
        )}
      </div>

      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum mb-6">
        <span className="text-ink">{list.total.toLocaleString()}</span>{" "}
        {section === "active" ? "uploaded" : "in Deleted"}
      </p>

      {photos.some((photo) => photo.deletionStartedAt) && (
        <p className="text-sm text-slate mb-6">
          Some photos have started permanent deletion and cannot be restored.
          Select them and retry Delete forever to finish cleanup.
        </p>
      )}

      {list.isLoading ? (
        <div className={gridClass}>
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="h-full w-full rounded-xl row-span-2" />
          ))}
        </div>
      ) : photos.length === 0 ? (
        section === "active" ? (
          <NoPhotosYet eventId={event.id} />
        ) : (
          <EmptyTrash />
        )
      ) : (
        <div className={gridClass}>
          {photos.map((photo, i) => (
            <PhotoTile
              key={photo.id}
              photo={photo}
              index={i}
              selecting={selecting}
              selected={selected.has(photo.id)}
              onOpen={() => {
                if (photo.deletionStartedAt) {
                  setSelecting(true);
                  toggle(photo.id);
                } else setOpenIndex(i);
              }}
              onToggle={() => toggle(photo.id)}
            />
          ))}
        </div>
      )}

      {photos.length > 0 && (
        <LoadMoreButton
          shown={photos.length}
          total={list.total}
          increment={PAGE_SIZE.PHOTO_INCREMENT}
          onLoadMore={list.fetchNextPage}
          isLoading={list.isFetchingNextPage}
        />
      )}

      {selecting && (
        <SelectionBar
          section={section}
          count={selected.size}
          restoreCount={photos.filter((photo) => selected.has(photo.id) && !photo.deletionStartedAt).length}
          allSelected={photos.length > 0 && selected.size === photos.length}
          busy={busy || downloading}
          downloading={downloading}
          onDownload={() => handleDownload([...selected])}
          onToggleAll={toggleAll}
          onTrash={() => runTrashOrRestore("trash")}
          onRestore={() => runTrashOrRestore("restore")}
          onPermanent={runPermanent}
          onDone={exitSelect}
        />
      )}

      {!selecting && previewItem && openIndex !== null && (
        <PhotoPreviewCard
          mode="owned"
          metadataAccess="owner"
          photo={previewItem}
          eventName={event.name}
          eventDate={event.date}
          index={openIndex + 1}
          total={photos.length}
          onClose={handleClose}
          onPrev={openIndex > 0 ? handlePrev : undefined}
          onNext={openIndex < photos.length - 1 ? handleNext : undefined}
          onDownload={() => handleDownloadSingle(previewItem.id)}
        />
      )}
    </section>
  );
}

function PurchasedGallery({ eventId }: { eventId: string }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const list = usePhotographerPurchasedPhotos(eventId, true);
  const [selection, setSelection] = useState<PurchaseSelection | null>(null);
  const photos = list.items.map((item) => item.photo);

  useEffect(() => {
    if (searchParams.get("tab") !== "purchased") {
      router.replace(`${ROUTES.DASHBOARD_EVENTS}/${eventId}?tab=purchased`, { scroll: false });
    }
  }, [eventId, router, searchParams]);

  return <>
    <p className="font-mono uppercase tracking-[0.14em] text-sm text-slate-soft tnum mb-6"><span className="text-ink">{list.total.toLocaleString()}</span> purchased photos</p>
    {list.isLoading ? (
      <div role="status" className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6">
        {Array.from({ length: 8 }, (_, i) => <TileSkeleton key={i} aspectRatio="aspect-[3/4]" />)}
      </div>
    ) : list.error ? (
      <div role="alert"><p>Couldn&apos;t load purchased photos.</p><button type="button" onClick={() => void list.refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-4")}>Retry</button></div>
    ) : photos.length === 0 ? (
      <p className="font-sans text-base text-slate">No purchased photos yet.</p>
    ) : (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 md:gap-6 grid-flow-row-dense [grid-auto-rows:96px] md:[grid-auto-rows:140px] lg:[grid-auto-rows:180px]">
        {list.items.map((item, i) => <PhotoTile key={item.photo.id} photo={item.photo} index={i} purchaseCount={item.purchaseCount} selecting={false} selected={false} onOpen={() => setSelection({ photoId: item.photo.id })} onToggle={() => { }} />)}
      </div>
    )}
    {list.appendError && <p role="alert" className="mt-4 font-sans text-sm text-slate">Couldn&apos;t load more purchased photos. Try again.</p>}
    {list.total > 0 && <LoadMoreButton shown={photos.length} total={list.total} increment={PAGE_SIZE.PHOTO_INCREMENT} onLoadMore={list.fetchNextPage} isLoading={list.isFetchingNextPage} />}
    <PhotographerPurchasePreview selection={selection} photos={photos} onClose={() => setSelection(null)} onSelectPhoto={(photoId) => setSelection({ photoId })} />
  </>;
}

function TabPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-4 py-2 min-h-[44px] transition-colors",
        active
          ? "bg-ink text-surface border border-ink"
          : "bg-bone text-slate border border-line hover:text-ink hover:border-ink",
      )}
    >
      {children}
    </button>
  );
}

function EmptyTrash() {
  return (
    <section className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        Nothing in Deleted.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Photos you delete land here first. Restore them any time, or remove them
        for good once you&apos;re sure.
      </p>
    </section>
  );
}

// Sticky bulk-action bar for select mode. Mirrors AdminPayoutBulkBar's recipe;
// on the Deleted tab, Restore is the one fresh accent while the bar is up.
function SelectionBar({
  section,
  count,
  restoreCount,
  allSelected,
  busy,
  downloading,
  onDownload,
  onToggleAll,
  onTrash,
  onRestore,
  onPermanent,
  onDone,
}: {
  section: "active" | "deleted";
  count: number;
  restoreCount: number;
  allSelected: boolean;
  busy: boolean;
  downloading: boolean;
  onDownload: () => void;
  onToggleAll: () => void;
  onTrash: () => void;
  onRestore: () => void;
  onPermanent: () => void;
  onDone: () => void;
}) {
  return (
    <div
      role="region"
      aria-label="Bulk photo actions"
      className="fixed inset-x-4 bottom-4 lg:left-auto lg:right-8 lg:bottom-6 lg:max-w-xl z-40 rounded-2xl border border-ink bg-bone shadow-2xl px-4 md:px-6 py-4 animate-[fade-up_0.2s_ease-out_both]"
    >
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-3 min-w-0">
          <Kicker as="p">
            <span className="tnum">{count}</span> selected
          </Kicker>
          <button
            type="button"
            onClick={onToggleAll}
            className={cn(BTN_GHOST, BTN_SIZE.sm, "px-2")}
          >
            {allSelected ? "Clear" : "Select all"}
          </button>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          {section === "active" ? (
            <>
              <button
                type="button"
                onClick={onDownload}
                disabled={busy || count === 0}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm, "tnum")}
              >
                {downloading ? "Preparing…" : `Download ${count}`}
              </button>
              <button
                type="button"
                onClick={onTrash}
                disabled={busy || count === 0}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                {busy ? "Working…" : `Delete ${count}`}
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                onClick={onRestore}
                disabled={busy || restoreCount === 0}
                className={cn(
                  BTN_PRIMARY,
                  BTN_SIZE.sm,
                  "disabled:hover:bg-fresh",
                )}
              >
                {busy ? "Working…" : `Restore ${restoreCount}`}
              </button>
              <button
                type="button"
                onClick={onPermanent}
                disabled={busy || count === 0}
                className={cn(BTN_DANGER, BTN_SIZE.sm)}
              >
                Delete forever
              </button>
            </>
          )}
          <button
            type="button"
            onClick={onDone}
            className={cn(BTN_GHOST, BTN_SIZE.sm, "px-3")}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}

function PhotoTile({
  photo,
  index,
  selecting,
  selected,
  onOpen,
  onToggle,
  purchaseCount,
}: {
  photo: PhotographerLibraryPhoto;
  index: number;
  selecting: boolean;
  selected: boolean;
  onOpen: () => void;
  onToggle: () => void;
  purchaseCount?: number;
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
        onClick={selecting ? onToggle : onOpen}
        aria-haspopup={selecting ? undefined : "dialog"}
        aria-pressed={selecting ? selected : undefined}
        aria-label={
          (selecting
            ? `${selected ? "Deselect" : "Select"} photo ${index + 1}`
            : `Open photo ${index + 1}${purchaseCount !== undefined ? `, ${purchaseCount} purchase${purchaseCount === 1 ? "" : "s"}` : ""}`) +
          (photo.status === "processing" ? ", Processing" : "")
        }
        className="block w-full h-full text-left rounded-xl focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        {/* Tone-color background acts as a placeholder while the watermarked
            JPEG decodes — once `thumbnailUrl` loads, object-cover fills the
            entire tile and the placeholder is hidden behind it. */}
        <div
          className={cn(
            "relative overflow-hidden rounded-xl h-full transition-transform duration-300",
            !selecting && "group-hover:-translate-y-[2px]",
            selecting &&
              selected &&
              "ring-2 ring-ink ring-offset-2 ring-offset-bone",
          )}
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
          {selecting && (
            <span
              aria-hidden="true"
              className={cn(
                "absolute top-2 left-2 size-6 rounded-full border flex items-center justify-center transition-colors",
                selected
                  ? "bg-ink border-ink text-surface"
                  : "bg-bone/70 border-line text-transparent backdrop-blur-sm",
              )}
            >
              <svg
                viewBox="0 0 20 20"
                className="size-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="3"
              >
                <path
                  d="M4 10l4 4 8-9"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </span>
          )}
          {photo.status === "processing" && (
            <span className="absolute bottom-2 left-2 rounded-full bg-bone px-2 py-1 font-mono text-[12px] uppercase tracking-wide text-ink">
              Processing
            </span>
          )}
          {(purchaseCount !== undefined || photo.salesCount > 0) && (
            <span className="absolute bottom-2 right-2 font-mono uppercase tracking-[0.14em] text-[12px] text-surface tnum bg-ink/70 px-2 py-1 rounded-full backdrop-blur-sm">
              {purchaseCount !== undefined ? `${purchaseCount} purchase${purchaseCount === 1 ? "" : "s"}` : `${photo.salesCount} sold`}
            </span>
          )}
        </div>
      </button>
    </div>
  );
}
