"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { SiteHeader } from "@/components/layout/site-header";
import { Slab } from "@/components/profile-shell";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { EventTile } from "@/components/events/event-tile";
import { isReservedHandle } from "@/lib/reserved-handles";
import { ROUTES } from "@/lib/constants";
import { formatMemberSince } from "@/lib/format";
import {
  PHOTOGRAPHER_EVENTS,
  type EventState,
} from "@/lib/photographer-mock";
import { EVENT_CATALOG } from "@/lib/event-catalog";
import {
  getPhotographerByHandle,
  getProfileTotals,
  type CoverSource,
  type PhotographerProfile,
} from "@/lib/photographer-registry";
import {
  usePhotographerSettingsStore,
  BRAND_COLOR_HEX,
} from "@/store/photographer-settings-store";
import type { ListEvent } from "@/app/events/events-browser";

// Public photographer profile. Pure read surface — no edit affordances, no
// URL pill, no "back to QuickPitik" footer, no Sales stat, even when the
// logged-in photographer visits their own /{handle}. Owner edit/share lives
// on /profile; this page is what runners see.
//
// Resolves three states:
//   1. Handle matches the logged-in photographer's settings store → render a
//      live preview built from the store + PHOTOGRAPHER_EVENTS.
//   2. Handle matches a static registry photographer → render their public
//      profile.
//   3. Otherwise → "No photographer here yet" not-found body.
//
// TODO(backend): replace registry lookup + settings-store override with a
// single `api.get<PhotographerPublic>("/p/{handle}")` call.
export default function PublicHandlePage() {
  const params = useParams<{ handle: string }>();
  const raw = Array.isArray(params.handle) ? params.handle[0] : params.handle;
  const handle = (raw ?? "").trim().toLowerCase();

  if (isReservedHandle(handle)) {
    notFound();
  }

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />
      <PublicProfileBody handle={handle} />
    </main>
  );
}

function PublicProfileBody({ handle }: { handle: string }) {
  const settings = usePhotographerSettingsStore();
  const isOwnHandle = handle.length > 0 && settings.handle === handle;

  if (isOwnHandle) {
    const profile = ownerProfileFromSettings(handle, settings);
    return <ProfileLayout profile={profile} isOwner />;
  }

  const registryProfile = getPhotographerByHandle(handle);
  if (registryProfile) {
    return <ProfileLayout profile={registryProfile} isOwner={false} />;
  }

  return <NotFoundBody handle={handle} />;
}

function ownerProfileFromSettings(
  handle: string,
  s: ReturnType<typeof usePhotographerSettingsStore.getState>,
): PhotographerProfile {
  return {
    handle,
    displayName: s.brandName.trim() || "Photographer",
    brandColor: s.brandColor,
    bio: s.bio,
    city: "Cebu",
    memberSince: "2024-08-15",
    cover: s.cover ? { kind: "image", url: s.cover.dataUrl } : null,
    watermarkLabel: s.watermark ? "OWNER" : null,
    events: PHOTOGRAPHER_EVENTS.filter((e) => e.photoCount > 0).map((e) => ({
      eventSlug: e.slug,
      state: e.state,
      photoCount: e.photoCount,
      salesCount: e.salesCount,
    })),
  };
}

/* ─────────────── LAYOUT ─────────────── */

function ProfileLayout({
  profile,
  isOwner,
}: {
  profile: PhotographerProfile;
  isOwner: boolean;
}) {
  const accent =
    profile.brandColor !== "none"
      ? BRAND_COLOR_HEX[profile.brandColor]
      : null;
  const totals = getProfileTotals(profile);
  const portfolioEvents = profile.events.filter((e) => e.photoCount > 0);

  return (
    <div className="flex-1">
      <CoverBanner cover={profile.cover} accent={accent} />

      {/* Body container matches SiteHeader (max-w-7xl + px-6 md:px-10) so the
          left/right edges line up with the QuickPitik wordmark and the
          Events / Profile chips in the header. */}
      <div className="max-w-7xl mx-auto w-full px-6 md:px-10 -mt-10 md:-mt-14 relative">
        <div className="flex flex-col items-start gap-4 md:flex-row md:items-end md:gap-5">
          <AvatarDisc
            name={profile.displayName}
            size="lg"
            avatarOverride={null}
          />
          <div className="flex-1 min-w-0 md:pb-2">
            <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
              Photographer
            </p>
          </div>
        </div>

        <div className="mt-6 flex items-baseline gap-3 flex-wrap">
          <h1 className="font-display text-4xl md:text-5xl font-medium tracking-tight text-ink leading-[1.05]">
            {profile.displayName}
          </h1>
          {accent && (
            <span
              aria-hidden="true"
              className="size-3 rounded-full inline-block shrink-0"
              style={{ backgroundColor: accent }}
            />
          )}
        </div>

        {profile.bio && (
          <p className="font-sans text-base md:text-lg text-ink-soft mt-3 max-w-xl leading-relaxed">
            {profile.bio}
          </p>
        )}

        <StatsRow totals={totals} memberSince={profile.memberSince} />
      </div>

      {portfolioEvents.length === 0 ? (
        <ComingSoonPanel displayName={profile.displayName} />
      ) : (
        <PortfolioSection
          events={portfolioEvents}
          handle={profile.handle}
          isOwner={isOwner}
        />
      )}
    </div>
  );
}

/* ─────────────── COVER ─────────────── */

function CoverBanner({
  cover,
  accent,
}: {
  cover: CoverSource | null;
  accent: string | null;
}) {
  return (
    <div className="relative bg-bone-deep border-b border-line aspect-[16/5] md:aspect-[16/4] overflow-hidden">
      {cover === null ? (
        <div
          aria-hidden="true"
          className="size-full"
          style={{
            background: `linear-gradient(135deg, var(--bone-deep), var(--bone))`,
          }}
        />
      ) : cover.kind === "image" ? (
        // Responsive fit: mobile fills the cover edge-to-edge (object-cover,
        // the photo dominates the small viewport), desktop+ contains the
        // full image (object-contain, bone-deep bands fill any empty space
        // — Facebook-style). The container's bg-bone-deep IS the band fill.
        // eslint-disable-next-line @next/next/no-img-element -- data URL preview; real cover comes from signed S3 URL.
        <img
          src={cover.url}
          alt=""
          className="size-full object-cover md:object-contain"
          draggable={false}
        />
      ) : (
        <div
          aria-hidden="true"
          className="size-full"
          style={{
            background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
          }}
        />
      )}
      {accent && (
        <span
          aria-hidden="true"
          className="absolute bottom-0 inset-x-0 h-1"
          style={{ backgroundColor: accent }}
        />
      )}
    </div>
  );
}

/* ─────────────── STATS ─────────────── */

// Public profile shows Events / Photos / On QuickPitik only. "Sales" is a
// photographer-side number — it stays on /profile and /dashboard/earnings
// where it belongs.
function StatsRow({
  totals,
  memberSince,
}: {
  totals: { events: number; photos: number; sales: number };
  memberSince: string;
}) {
  return (
    <dl className="mt-10 grid grid-cols-3 gap-y-7 gap-x-8 md:gap-x-12 max-w-2xl">
      <Stat label="Events" value={totals.events.toString()} />
      <Stat label="Photos" value={totals.photos.toLocaleString()} />
      <Stat label="On QuickPitik" value={formatMemberSince(memberSince)} />
    </dl>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
        {label}
      </p>
      <p className="font-display font-medium tracking-tight tnum text-2xl md:text-3xl text-ink mt-2 leading-none">
        {value}
      </p>
    </div>
  );
}

/* ─────────────── PORTFOLIO ─────────────── */

// Reuses <EventTile mode="browse"> from the public /events listing so the
// visual language matches across the site. `hrefOverride` redirects the
// click into the per-photographer gallery (/{handle}/events/{slug}) instead
// of the runner-wide cockpit (/events/{slug}) — when you're on Paksit's
// profile, clicking should drop you into Paksit's photos for that event,
// not the multi-photographer cockpit.
function PortfolioSection({
  events,
  handle,
  isOwner,
}: {
  events: ReadonlyArray<{
    eventSlug: string;
    state: EventState;
    photoCount: number;
    salesCount: number;
  }>;
  handle: string;
  isOwner: boolean;
}) {
  return (
    <div className="max-w-7xl mx-auto w-full px-6 md:px-10 mt-10 md:mt-14 pb-20">
      <Slab
        id="portfolio"
        number="01"
        title="Portfolio"
        trailing={`${events.length} ${events.length === 1 ? "event" : "events"}`}
      >
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
          {events.map((coverage, index) => {
            const tileEvent = toListEvent(coverage);
            if (!tileEvent) return null;
            return (
              <EventTile
                key={coverage.eventSlug}
                mode="browse"
                event={tileEvent}
                index={index}
                hrefOverride={`/${handle}/events/${coverage.eventSlug}`}
                saveDisabled={isOwner}
              />
            );
          })}
        </div>
      </Slab>
    </div>
  );
}

// Build the ListEvent shape EventTile expects from a photographer's coverage
// row plus the canonical event in EVENT_CATALOG. We override `photoCount`
// with the photographer's slice (e.g. Paksit shot 412 of Cebu Marathon's
// 1,200 photos) and `state` with the coverage state — everything else
// (id, slug, name, date, location, city, participantCount, status) flows
// from the catalog.
function toListEvent(coverage: {
  eventSlug: string;
  state: EventState;
  photoCount: number;
}): ListEvent | null {
  const ce = EVENT_CATALOG.find((e) => e.slug === coverage.eventSlug);
  if (!ce) return null;
  return {
    ...ce,
    photoCount: coverage.photoCount,
    state: coverage.state,
  };
}

/* ─────────────── EMPTY STATE ─────────────── */

function ComingSoonPanel({ displayName }: { displayName: string }) {
  return (
    <section className="max-w-7xl mx-auto w-full px-6 md:px-10 mt-12 md:mt-16 pb-20">
      <div className="border border-dashed border-line rounded-2xl p-10 md:p-16 text-center">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Portfolio
        </p>
        <p className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink mt-4">
          Photos coming soon.
        </p>
        <p className="font-sans text-base text-ink-soft mt-3 max-w-md mx-auto">
          Once {displayName} uploads their first event, every photo they shoot
          will land here — searchable by face or bib.
        </p>
      </div>
    </section>
  );
}

/* ─────────────── NOT FOUND ─────────────── */

function NotFoundBody({ handle }: { handle: string }) {
  return (
    <div className="flex-1 flex items-center justify-center px-6 md:px-10 py-20">
      <div className="text-center max-w-md">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Not found
        </p>
        <h1 className="font-display text-4xl md:text-5xl font-medium tracking-tight text-ink mt-4">
          No photographer here yet.
        </h1>
        <p className="font-sans text-base text-ink-soft mt-3">
          {handle.length > 0 ? (
            <>
              We couldn&apos;t find{" "}
              <span className="font-mono">quickpitik.com/{handle}</span>. They
              might have changed their URL.
            </>
          ) : (
            <>The handle you visited isn&apos;t taken.</>
          )}
        </p>
        <Link
          href={ROUTES.HOME}
          className="mt-6 inline-block font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
        >
          Back to QuickPitik
        </Link>
      </div>
    </div>
  );
}
