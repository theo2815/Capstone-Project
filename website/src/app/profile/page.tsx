"use client";

import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { AvatarDisc } from "@/components/account/avatar-disc";
import {
  IdentityRail,
  Slab,
  type JumpSection,
} from "@/components/profile-shell";
import { SelfieLibrary } from "@/components/profile/selfie-library";
import { EventTile } from "@/components/events/event-tile";
import {
  EventFilterBar,
  EventFilterEmpty,
  matchEventDate,
  type EventDateKey,
} from "@/components/dashboard/event-filter-bar";
import { Kicker } from "@/components/ui/kicker";
import { useAuth } from "@/hooks/use-auth";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useOrdersStore } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import {
  getCatalogWithOverrides,
  MOCK_USER_PHOTOS_FOUND,
} from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { formatMemberSince, formatRaceDate } from "@/lib/format";
import { PHOTOGRAPHER_EVENTS } from "@/lib/photographer-mock";
import type { ListEvent } from "@/app/events/events-browser";
import {
  usePhotographerSettingsStore,
  BRAND_COLOR_HEX,
} from "@/store/photographer-settings-store";
import type { Role, User } from "@/types/user";

type RaceState = "upcoming" | "live" | "open" | "past";

interface RaceLogEntry {
  id: string;
  slug: string;
  name: string;
  date: string;
  state: RaceState;
  photosFound: number;
  photosBought: number;
  isSaved: boolean;
}

const JUMP_SECTIONS: ReadonlyArray<JumpSection> = [
  { id: "selfies", label: "Selfie library" },
  { id: "race-log", label: "Race log" },
];

export default function ProfilePage() {
  return (
    <ProtectedRoute>
      <ProfileRouter />
    </ProtectedRoute>
  );
}

// Role-branched: runners see the selfie library + race log shell; photographers
// see a cover-banner-led portfolio surface that mirrors their public /{handle}
// page with edit affordances.
function ProfileRouter() {
  const { user } = useAuth();
  if (!user) return null;
  if (user.role === "PHOTOGRAPHER") return <PhotographerProfileBody user={user} />;
  return <RunnerProfileBody user={user} />;
}

// ─────────────────────────────────────────────────────────────────────────
// Runner profile (existing layout)
// ─────────────────────────────────────────────────────────────────────────

function RunnerProfileBody({ user }: { user: User }) {
  const memberSince = formatMemberSince(user.createdAt);

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 flex flex-col">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20 flex-1">
          <IdentityRail
            user={user}
            kicker={
              <>
                {roleLabel(user.role)}
                {" · Cebu · "}
                <span className="tnum">Since {memberSince}</span>
              </>
            }
            headline={user.name}
            headlineSize="name"
            subline={<span className="break-all">{user.email}</span>}
            jumpSections={JUMP_SECTIONS}
            currentPath={ROUTES.PROFILE}
          />
          <div className="stagger-children min-w-0 pb-8 md:pb-20 md:border-l md:border-line md:-ml-6 lg:-ml-10 md:pl-6 lg:pl-10">
            <SelfieLibrarySection />
            <RaceLogSection />
          </div>
        </div>
      </div>
    </main>
  );
}

function roleLabel(role: Role): string {
  if (role === "RUNNER") return "Runner";
  if (role === "PHOTOGRAPHER") return "Photographer";
  return "Admin";
}

function SelfieLibrarySection() {
  return (
    <Slab
      id="selfies"
      number="01"
      title="Selfie library"
      caption="Used by face search across every event"
    >
      <SelfieLibrary />
    </Slab>
  );
}

function RaceLogSection() {
  const savedIds = useSavedEventsStore((s) => s.ids);
  const orders = useOrdersStore((s) => s.orders);

  const log = useMemo(
    () => buildRaceLog(savedIds, orders),
    [savedIds, orders],
  );

  const trailing = `${log.length} race${log.length === 1 ? "" : "s"}`;
  return (
    <Slab id="race-log" number="02" title="Race log" trailing={trailing}>
      {log.length === 0 ? (
        <RaceLogEmpty />
      ) : (
        <ul className="border-y border-line divide-y divide-line">
          {log.map((entry) => (
            <li key={entry.id}>
              <RaceLogRow entry={entry} />
            </li>
          ))}
        </ul>
      )}
    </Slab>
  );
}

function RaceLogRow({ entry }: { entry: RaceLogEntry }) {
  const unsave = useSavedEventsStore((s) => s.unsave);
  const save = useSavedEventsStore((s) => s.save);
  const { showToast } = useToast();
  const date = formatRaceDate(entry.date);
  const isUpcoming = entry.state === "upcoming";
  const showUnsave = isUpcoming && entry.isSaved;
  const stateLabel: Record<RaceState, string> = {
    upcoming: "Saved",
    live: "Photos uploading",
    open: "Photos ready",
    past: "Archived",
  };

  const body = (
    <div className="flex items-baseline justify-between gap-6 py-6 md:py-7">
      <div className="flex-1 min-w-0">
        <Kicker as="p" tnum>
          {date}
        </Kicker>
        <h3 className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 truncate">
          {entry.name}
        </h3>
        <p className="font-sans text-sm text-slate mt-2">
          {stateLabel[entry.state]}
          {!isUpcoming && entry.photosFound > 0 && (
            <>
              <span className="text-slate-soft"> · </span>
              <span className="font-mono tnum text-ink-soft">
                {entry.photosFound}
              </span>{" "}
              photos
            </>
          )}
          {entry.photosBought > 0 && (
            <>
              <span className="text-slate-soft"> · </span>
              <span className="font-mono tnum text-fresh">
                {entry.photosBought}
              </span>{" "}
              <span className="text-fresh">kept</span>
            </>
          )}
        </p>
      </div>
      {showUnsave ? (
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            // TODO(backend): swap for `api.delete("/me/saved-events/${entry.id}")`.
            unsave(entry.id);
            showToast({
              kind: "success",
              message: `Removed ${entry.name} from saved.`,
              action: {
                label: "Undo",
                onClick: () => save(entry.id),
              },
            });
          }}
          aria-label={`Unsave ${entry.name}`}
          className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors shrink-0"
        >
          Unsave
        </button>
      ) : !isUpcoming ? (
        <span className="font-sans text-sm text-ink group-hover:text-fresh transition-colors shrink-0 inline-flex items-center gap-1.5">
          Open
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </span>
      ) : null}
    </div>
  );

  if (isUpcoming) {
    return <div className="group block">{body}</div>;
  }

  return (
    <Link
      href={`/events/${entry.slug}`}
      aria-label={`Open ${entry.name}`}
      className="group block rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      {body}
    </Link>
  );
}

function RaceLogEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No races yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Save a race or buy a photo to start your log.
      </p>
      <Link
        href={ROUTES.EVENTS}
        className="mt-6 inline-block font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
      >
        Browse races
      </Link>
    </div>
  );
}

// Race Log = saved ∪ purchased. Saved events appear immediately. Purchases
// retroactively create rows when the user bought from an event they didn't save.
// Both triggers can apply to the same event — single row, dedupe by event id.
function buildRaceLog(
  savedIds: ReadonlyArray<string>,
  orders: ReadonlyArray<{ eventId: string; photoIds: string[] }>,
): ReadonlyArray<RaceLogEntry> {
  const photosByEvent = new Map<string, number>();
  for (const order of orders) {
    photosByEvent.set(
      order.eventId,
      (photosByEvent.get(order.eventId) ?? 0) + order.photoIds.length,
    );
  }

  const includedIds = new Set<string>([
    ...savedIds,
    ...photosByEvent.keys(),
  ]);

  const entries: RaceLogEntry[] = [];
  for (const id of includedIds) {
    const event = getCatalogWithOverrides().find((e) => e.id === id);
    if (!event) continue;
    entries.push({
      id: event.id,
      slug: event.slug,
      name: event.name,
      date: event.date,
      state: event.state,
      photosFound: MOCK_USER_PHOTOS_FOUND[event.id] ?? 0,
      photosBought: photosByEvent.get(event.id) ?? 0,
      isSaved: savedIds.includes(event.id),
    });
  }

  return entries.sort((a, b) => b.date.localeCompare(a.date));
}

// ─────────────────────────────────────────────────────────────────────────
// Photographer profile (cover banner + brand + portfolio)
// ─────────────────────────────────────────────────────────────────────────

function PhotographerProfileBody({ user }: { user: User }) {
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const brandColor = usePhotographerSettingsStore((s) => s.brandColor);
  const bio = usePhotographerSettingsStore((s) => s.bio);
  const handle = usePhotographerSettingsStore((s) => s.handle);

  const accent = brandColor !== "none" ? BRAND_COLOR_HEX[brandColor] : null;
  const displayName = brandName.trim() || user.name;

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col">
      <SiteHeader />

      {/* Cover banner — full-bleed inside the SiteHeader's max width.
          Responsive fit: mobile fills the cover edge-to-edge (object-cover,
          the photo dominates the small viewport), desktop+ contains the
          full image (object-contain, bone-deep bands fill any empty space
          — Facebook-style). The container's bg-bone-deep IS the band fill. */}
      <div className="relative bg-bone-deep border-b border-line aspect-[16/5] md:aspect-[16/4] overflow-hidden">
        {cover ? (
          // eslint-disable-next-line @next/next/no-img-element -- data-URL mock; backend will return signed S3 URLs.
          <img
            src={cover.dataUrl}
            alt=""
            className="size-full object-cover md:object-contain"
            draggable={false}
          />
        ) : (
          <CoverEmptyHint />
        )}
        {accent && (
          <span
            aria-hidden="true"
            className="absolute bottom-0 inset-x-0 h-1"
            style={{ backgroundColor: accent }}
          />
        )}
      </div>

      <div className="flex-1 max-w-7xl mx-auto w-full px-6 md:px-10 -mt-10 md:-mt-12 relative">
        {/* Edit cover — top-right of the negative-margin band. Mobile: own
            right-aligned row above the avatar. Desktop: absolute, free of
            the avatar/kicker row. Routes to settings #cover anchor. */}
        <div className="flex justify-end mb-3 md:mb-0 md:absolute md:top-0 md:right-10 md:z-10">
          <Link
            href={`${ROUTES.DASHBOARD_SETTINGS}#cover`}
            className="inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors"
          >
            <CameraGlyph className="size-4" />
            Edit cover
            <span aria-hidden="true">→</span>
          </Link>
        </div>

        <div className="flex flex-col items-start gap-4 md:flex-row md:items-end md:gap-5">
          <AvatarDisc name={displayName} size="lg" />
          <div className="flex-1 min-w-0 md:pb-2">
            <Kicker as="p">
              Photographer
            </Kicker>
          </div>
        </div>

        <div className="mt-6 flex items-baseline gap-3 flex-wrap">
          <h1 className="font-display text-4xl md:text-5xl font-medium tracking-tight text-ink leading-[1.05]">
            {displayName}
          </h1>
          {accent && (
            <span
              aria-hidden="true"
              className="size-3 rounded-full inline-block"
              style={{ backgroundColor: accent }}
            />
          )}
        </div>

        {bio && (
          <p className="font-sans text-base md:text-lg text-ink-soft mt-3 max-w-xl">
            {bio}
          </p>
        )}

        {/* Quick actions — Upload (primary fresh) + Edit profile (text-link). */}
        <div className="mt-7 flex flex-col sm:flex-row sm:items-center gap-4 sm:gap-6">
          <Link
            href={ROUTES.DASHBOARD_UPLOAD}
            className="inline-flex items-center justify-center gap-2 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors w-full sm:w-auto"
          >
            <span aria-hidden="true">+</span>
            Upload photos
          </Link>
          <Link
            href={ROUTES.DASHBOARD_SETTINGS}
            className="font-sans text-sm text-ink hover:text-fresh transition-colors inline-flex items-center gap-1 self-start sm:self-auto"
          >
            Edit profile
            <span aria-hidden="true">→</span>
          </Link>
        </div>

        <StatsRow memberSince={user.createdAt} />

        <PublicUrlChip handle={handle} />

        <div className="stagger-children mt-12 pb-8 md:pb-20">
          <PortfolioGrid handle={handle} />
        </div>
      </div>

    </main>
  );
}

// Photographer-self stats: Events / Photos / Sales / On QuickPitik. Mirrors
// the public /[handle] stats row but adds Sales (which is owner-relevant).
// Numbers derived from PHOTOGRAPHER_EVENTS rather than the public profile's
// `getProfileTotals` since /profile reads the live mock list directly.
function StatsRow({ memberSince }: { memberSince: string }) {
  const events = PHOTOGRAPHER_EVENTS.filter((e) => e.photoCount > 0);
  const totalPhotos = events.reduce((sum, e) => sum + e.photoCount, 0);
  const totalSales = events.reduce((sum, e) => sum + e.salesCount, 0);
  return (
    <dl className="mt-10 grid grid-cols-2 md:grid-cols-4 gap-y-7 gap-x-8 md:gap-x-12 max-w-2xl">
      <Stat label="Events" value={events.length.toString()} />
      <Stat label="Photos" value={totalPhotos.toLocaleString()} />
      <Stat label="Sales" value={totalSales.toLocaleString()} />
      <Stat label="On QuickPitik" value={formatMemberSince(memberSince)} />
    </dl>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <Kicker as="p">
        {label}
      </Kicker>
      <p className="font-display font-medium tracking-tight tnum text-2xl md:text-3xl text-ink mt-2 leading-none">
        {value}
      </p>
    </div>
  );
}

function CoverEmptyHint() {
  return (
    <div className="absolute inset-0 flex items-center justify-center">
      <div className="text-center">
        <Kicker as="p" tone="soft">
          No cover yet
        </Kicker>
        <Link
          href={ROUTES.DASHBOARD_SETTINGS}
          className="mt-2 inline-block font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
        >
          Add a banner →
        </Link>
      </div>
    </div>
  );
}

function PublicUrlChip({ handle }: { handle: string }) {
  const { showToast } = useToast();
  const valid = handle.length > 0;
  const url = valid
    ? `quickpitik.com/${handle}`
    : "quickpitik.com/your-handle";

  const copy = useCallback(async () => {
    if (!valid) return;
    try {
      await navigator.clipboard.writeText(`https://${url}`);
      showToast({ kind: "success", message: "Public URL copied." });
    } catch {
      showToast({
        kind: "error",
        message: "Couldn't copy. Try selecting the URL manually.",
      });
    }
  }, [url, valid, showToast]);

  return (
    <div className="mt-6 border border-line rounded-2xl bg-bone-deep/40 px-5 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-3 md:gap-6">
      <div className="min-w-0">
        <Kicker as="p" tone="soft">
          Public URL
        </Kicker>
        <p className="font-mono text-sm md:text-base text-ink mt-1.5 break-all">
          {url}
        </p>
      </div>
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 shrink-0">
        <button
          type="button"
          onClick={copy}
          disabled={!valid}
          className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:no-underline"
        >
          Copy
        </button>
        <Link
          href={valid ? `/${handle}` : "#"}
          target="_blank"
          rel="noopener noreferrer"
          aria-disabled={!valid}
          onClick={(e) => {
            if (!valid) e.preventDefault();
          }}
          className={
            valid
              ? "font-sans text-sm text-ink hover:text-fresh transition-colors inline-flex items-center gap-1"
              : "font-sans text-sm text-slate-soft cursor-not-allowed inline-flex items-center gap-1"
          }
        >
          Preview public
          <span aria-hidden="true">↗</span>
        </Link>
      </div>
    </div>
  );
}

// Portfolio grid — reuses <EventTile mode="browse"> exactly like /events and
// /[handle], so "what I see is what runners see" on this surface. Each tile
// routes to /{handle}/events/{slug} (the photographer's own gallery) via
// hrefOverride; falls back to /events/{slug} only if the photographer hasn't
// claimed a handle yet. Date dropdown + name search reuse the /events
// taxonomy at the surface level.
function PortfolioGrid({ handle }: { handle: string }) {
  const allEvents = useMemo(
    () => PHOTOGRAPHER_EVENTS.filter((e) => e.photoCount > 0),
    [],
  );
  const validHandle = handle.length > 0;
  const [date, setDate] = useState<EventDateKey>("any");
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const trimmed = query.trim().toLowerCase();
    return allEvents.filter((summary) => {
      if (!matchEventDate(summary.state, date)) return false;
      if (trimmed) {
        const ce = getCatalogWithOverrides().find((e) => e.slug === summary.slug);
        if (!ce) return false;
        const hay = `${ce.name} ${ce.location} ${ce.city}`.toLowerCase();
        if (!hay.includes(trimmed)) return false;
      }
      return true;
    });
  }, [allEvents, date, query]);

  const clearFilters = () => {
    setDate("any");
    setQuery("");
  };

  return (
    <Slab
      id="portfolio"
      number="01"
      title="Portfolio"
      trailing={
        allEvents.length > 0
          ? `${allEvents.length} ${allEvents.length === 1 ? "event" : "events"}`
          : undefined
      }
    >
      {allEvents.length === 0 ? (
        <PortfolioEmpty />
      ) : (
        <>
          <EventFilterBar
            date={date}
            onDateChange={setDate}
            query={query}
            onQueryChange={setQuery}
            dateAriaLabel="Filter portfolio by date"
            searchAriaLabel="Search portfolio events"
          />
          {filtered.length === 0 ? (
            <EventFilterEmpty onClear={clearFilters} />
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 md:gap-8">
              {filtered.map((summary, index) => {
                const tileEvent = toListEvent(summary);
                if (!tileEvent) return null;
                return (
                  <EventTile
                    key={summary.id}
                    mode="browse"
                    event={tileEvent}
                    index={index}
                    hrefOverride={
                      validHandle
                        ? `/${handle}/events/${summary.slug}`
                        : undefined
                    }
                  />
                );
              })}
            </div>
          )}
        </>
      )}
    </Slab>
  );
}

// Build a ListEvent for EventTile from a PhotographerEventSummary. The
// catalog supplies city + participantCount + status (which the tile reads),
// the summary supplies the photographer's per-event state + their slice of
// photoCount.
function toListEvent(
  summary: (typeof PHOTOGRAPHER_EVENTS)[number],
): ListEvent | null {
  const ce = getCatalogWithOverrides().find((e) => e.slug === summary.slug);
  if (!ce) return null;
  return {
    ...ce,
    photoCount: summary.photoCount,
    state: summary.state,
  };
}

function PortfolioEmpty() {
  // Drops fresh button — the identity-row "Upload photos" CTA already owns
  // the page's single fresh element. Empty state points users back up.
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        Nothing shipped yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Cover an event and upload your first photos — they&apos;ll show up here
        as your portfolio.
      </p>
      <Link
        href={ROUTES.DASHBOARD_UPLOAD}
        className="mt-6 inline-flex items-center gap-1 font-sans text-sm text-ink hover:text-fresh transition-colors"
      >
        Upload photos
        <span aria-hidden="true">→</span>
      </Link>
    </div>
  );
}

function CameraGlyph({ className = "size-4" }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
    >
      <path
        d="M3 8h3.5l1.7-2.5h7.6L17.5 8H21v11H3V8z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
      <circle
        cx="12"
        cy="13.5"
        r="3.5"
        stroke="currentColor"
        strokeWidth="1.5"
      />
    </svg>
  );
}
