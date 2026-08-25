"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { Sparkline } from "@/components/dashboard/sparkline";
import { Skeleton, TileSkeleton } from "@/components/ui/skeleton";
import { useAuth } from "@/hooks/use-auth";
import {
  usePhotographerEarnings,
  usePhotographerEvents,
} from "@/hooks/use-photographer-data";
import { ROUTES } from "@/lib/constants";
import {
  BRAND_COLOR_HEX,
  usePhotographerSettingsStore,
} from "@/store/photographer-settings-store";

// 2×2 quick-action grid that anchors the dashboard overview. Each card
// carries a visual signature drawn from real photographer data — photo-tone
// strip / brand band + avatar / sparkline / event-state dot rows — so the
// tile pays its own way without a heavy stat block. Replaces the prior
// race-day cockpit hero.

const PHOTO_TONE_VARS = [
  "var(--ink)",
  "var(--ink-soft)",
  "var(--slate)",
  "var(--slate-soft)",
] as const;

export function DashboardActionGrid() {
  const { user } = useAuth();
  const firstName = user?.name?.split(/\s+/)[0] ?? "there";

  return (
    <section className="pb-12 md:pb-16">
      <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate tnum">
        Overview · Cebu
      </p>
      <h2 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink leading-[1.05] mt-4">
        Welcome back, {firstName}.
      </h2>

      <div className="mt-8 md:mt-10 grid grid-cols-1 sm:grid-cols-2 gap-4 md:gap-6 auto-rows-fr">
        <UploadCard />
        <ProfileCard />
        <EarningsCard />
        <EventsCard />
      </div>
    </section>
  );
}

function CardShell({
  href,
  children,
}: {
  href: string;
  children: ReactNode;
}) {
  return (
    <Link
      href={href}
      className="group flex flex-col h-full border border-line rounded-2xl p-5 md:p-6 bg-surface shadow-[var(--shadow-card)] hover:shadow-[var(--shadow-lift)] hover:border-fresh/40 hover:-translate-y-1 transition-all duration-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      {children}
    </Link>
  );
}

function CardLabel({ children }: { children: ReactNode }) {
  return (
    <p className="font-display text-xl md:text-2xl font-bold tracking-tight text-ink mt-auto">
      {children}
    </p>
  );
}

function CardCaption({
  children,
  arrowFresh = false,
}: {
  children: ReactNode;
  arrowFresh?: boolean;
}) {
  return (
    <p className="font-sans text-sm text-slate mt-2 inline-flex items-center gap-1.5">
      <span className="truncate">{children}</span>
      <span
        aria-hidden="true"
        className={
          "transition-transform group-hover:translate-x-0.5 shrink-0 " +
          (arrowFresh ? "text-fresh" : "text-ink")
        }
      >
        →
      </span>
    </p>
  );
}

// ─── Upload — photo-tone strip ────────────────────────────────────────────
function UploadCard() {
  return (
    <CardShell href={ROUTES.DASHBOARD_UPLOAD}>
      <div className="flex gap-1.5">
        {PHOTO_TONE_VARS.concat(PHOTO_TONE_VARS[0]).map((color, i) => (
          <div
            key={i}
            className="flex-1 h-10 md:h-12 rounded-md"
            style={{ backgroundColor: color }}
          />
        ))}
      </div>
      <CardLabel>
        <span className="text-fresh">+</span> Upload photos
      </CardLabel>
      <CardCaption arrowFresh>any Cebu race</CardCaption>
    </CardShell>
  );
}

// ─── Profile — brand-color band + avatar + display name ───────────────────
function ProfileCard() {
  const { user } = useAuth();
  const brandColor = usePhotographerSettingsStore((s) => s.brandColor);
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const handle = usePhotographerSettingsStore((s) => s.handle);

  if (!user) return null;

  const bandHex = BRAND_COLOR_HEX[brandColor];
  const bandStyle =
    bandHex === "transparent"
      ? { backgroundColor: "var(--bone-deep)" }
      : { backgroundColor: bandHex };
  const displayName = brandName.trim() || user.name;
  const handleLabel = handle.trim() ? `quickpitik.com/${handle}` : "set your public URL";

  return (
    <CardShell href={ROUTES.PROFILE}>
      <div>
        <div
          aria-hidden="true"
          className="h-1.5 rounded-full"
          style={bandStyle}
        />
        <div className="mt-4 md:mt-5 flex items-center gap-3 min-w-0">
          <AvatarDisc name={user.name} size="sm" />
          <p className="font-display text-base md:text-lg text-ink truncate">
            {displayName}
          </p>
        </div>
      </div>
      <CardLabel>Open profile</CardLabel>
      <CardCaption>{handleLabel}</CardCaption>
    </CardShell>
  );
}

// ─── Earnings — 12-week sparkline ─────────────────────────────────────────
// Reads GET /me/photographer/earnings directly — null while in flight, then
// the live numbers. No mock fallback: the previous `PHOTOGRAPHER_EARNINGS`
// seed made a BE 404 / slow start render fake ₱90k stats as the photographer's
// real earnings on the dashboard hero.
function EarningsCard() {
  const e = usePhotographerEarnings();
  if (!e) {
    return (
      <CardShell href={ROUTES.DASHBOARD_EARNINGS}>
        <TileSkeleton aspectRatio="h-16 md:h-20" className="rounded-md" />
        <CardLabel>View earnings</CardLabel>
        <Skeleton className="h-3 w-44 mt-2" />
      </CardShell>
    );
  }
  return (
    <CardShell href={ROUTES.DASHBOARD_EARNINGS}>
      <Sparkline
        data={e.weeklySeries}
        ariaLabel="Weekly earnings, last 12 weeks"
      />
      <CardLabel>View earnings</CardLabel>
      <CardCaption>
        ₱{e.thisWeek.toLocaleString()} this week · +{e.thisWeekSold} sold
      </CardCaption>
    </CardShell>
  );
}

// ─── Events — state-dot rows ──────────────────────────────────────────────
function EventsCard() {
  const events = usePhotographerEvents();
  const isLoading = events === null;
  if (isLoading || !events) {
    return (
      <CardShell href={ROUTES.DASHBOARD_EVENTS}>
        <ul className="space-y-2">
          <li>
            <Skeleton className="h-3 w-32" />
          </li>
          <li>
            <Skeleton className="h-3 w-24" />
          </li>
          <li>
            <Skeleton className="h-3 w-28" />
          </li>
          <li>
            <Skeleton className="h-3 w-36" />
          </li>
        </ul>
        <CardLabel>Manage events</CardLabel>
        <Skeleton className="h-3 w-36 mt-2" />
      </CardShell>
    );
  }
  // "open" is its own row rather than being folded into Archived. The two
  // mean opposite things to a photographer: an open event still accepts
  // uploads during the 4-day grace window, an archived one doesn't.
  const live = events.filter((e) => e.state === "live").length;
  const open = events.filter((e) => e.state === "open").length;
  const upcoming = events.filter((e) => e.state === "upcoming").length;
  const archived = events.filter((e) => e.state === "past").length;

  return (
    <CardShell href={ROUTES.DASHBOARD_EVENTS}>
      <ul className="space-y-2 font-mono uppercase tracking-[0.25em] text-[12px] text-slate tnum">
        <li className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className={
              live > 0
                ? "size-1.5 rounded-full bg-fresh breathe"
                : "size-1.5 rounded-full border border-line bg-bone"
            }
          />
          <span className="text-ink">{live}</span>
          <span>Live</span>
        </li>
        <li className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className="size-1.5 rounded-full bg-ink-soft"
          />
          <span className="text-ink">{open}</span>
          <span>Open</span>
        </li>
        <li className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className="size-1.5 rounded-full bg-slate"
          />
          <span className="text-ink">{upcoming}</span>
          <span>Upcoming</span>
        </li>
        <li className="flex items-center gap-2.5">
          <span
            aria-hidden="true"
            className="size-1.5 rounded-full border border-line bg-bone"
          />
          <span className="text-ink">{archived}</span>
          <span>Archived</span>
        </li>
      </ul>
      <CardLabel>Manage events</CardLabel>
      <CardCaption>your covered races</CardCaption>
    </CardShell>
  );
}
