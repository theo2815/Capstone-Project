"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { Slab } from "@/components/profile-shell";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { AdminActionAside } from "@/components/admin/admin-action-aside";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import {
  useAdminPhotographerView,
  syntheticCoverGradient,
} from "@/lib/admin-photographer-view";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatMemberSince } from "@/lib/format";
import { useEventCatalog } from "@/lib/event-catalog";
import { PAYOUT_METHOD_LABEL } from "@/store/photographer-settings-store";
import type { CoverSource } from "@/lib/photographer-registry";
import type { PhotographerSettingsSnapshot } from "@/lib/admin-user-registry";
import { formatPayoutNumber } from "@/lib/payout-format";

const COMPLETENESS_FIELDS: ReadonlyArray<{
  key: keyof PhotographerSettingsSnapshot;
  label: string;
}> = [
  { key: "hasCover", label: "Cover banner" },
  { key: "hasBrandName", label: "Brand name" },
  { key: "hasWatermark", label: "Watermark" },
  { key: "hasHandle", label: "Public handle" },
  { key: "hasRegion", label: "Region" },
];

const STATUS_LABEL = {
  approved: "Approved",
  pending: "Pending",
  incomplete: "Incomplete",
} as const;

// Phase 2a admin photographer detail. Hero + 5 content slabs (completeness,
// about, payouts, events covered, activity) + sticky-on-lg <AdminActionAside>.
// Reads via useAdminPhotographerView() so admin store overrides + live
// settings flow through to the visible state without a refresh.
export default function AdminPhotographerDetailPage() {
  const params = useParams<{ handle: string }>();
  const raw = Array.isArray(params.handle) ? params.handle[0] : params.handle;
  const handle = (raw ?? "").trim().toLowerCase();
  const view = useAdminPhotographerView(handle);
  const catalog = useEventCatalog();

  if (!view) {
    notFound();
  }

  const { row, profile, liveSettings, decisions } = view;
  const displayName = row.brandName ?? row.name;
  const cover: CoverSource =
    profile?.cover ??
    (liveSettings?.cover
      ? { kind: "image", url: liveSettings.cover.dataUrl }
      : { kind: "gradient", ...syntheticCoverGradient(row.userId) });
  const bio =
    liveSettings?.bio?.trim() || profile?.bio || "Bio not set yet.";
  const memberSinceIso =
    profile?.memberSince ?? row.createdAt;
  const isSuspended = row.suspendedAt !== null;
  const statusLabel = isSuspended
    ? "Suspended"
    : STATUS_LABEL[row.verificationStatus];
  const statusTone: "fresh" | "ink" | "slate" = isSuspended
    ? "ink"
    : row.verificationStatus === "approved"
      ? "fresh"
      : "slate";

  return (
    <>
      <div className="pb-2">
        <Link
          href={ROUTES.ADMIN_PHOTOGRAPHERS}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
        >
          ← Photographers
        </Link>
      </div>

      <Hero
        cover={cover}
        displayName={displayName}
        handle={row.handle}
        statusLabel={statusLabel}
        statusTone={statusTone}
        memberSince={memberSinceIso}
        region={row.region}
        city={row.city}
        decisionCount={decisions.length}
      />

      <div className="mt-10 lg:mt-14 grid lg:grid-cols-[1fr_18rem] lg:gap-12 lg:items-start">
        <div>
          <CompletenessSlab
            snapshot={row.settingsSnapshot}
            liveSettings={liveSettings}
          />
          <AboutSlab
            bio={bio}
            region={row.region}
            city={row.city}
            memberSince={memberSinceIso}
            email={row.email}
          />
          <PayoutsSlab liveSettings={liveSettings} />
          <EventsCoveredSlab
            events={profile?.events ?? []}
            catalogResolver={(slug) =>
              catalog.find((e) => e.slug === slug)?.name
            }
          />
          <ActivitySlab userId={row.userId} />
        </div>
        <AdminActionAside row={row} />
      </div>
    </>
  );
}

/* ─────────────── HERO ─────────────── */

function Hero({
  cover,
  displayName,
  handle,
  statusLabel,
  statusTone,
  memberSince,
  region,
  city,
  decisionCount,
}: {
  cover: CoverSource;
  displayName: string;
  handle: string | null;
  statusLabel: string;
  statusTone: "fresh" | "ink" | "slate";
  memberSince: string;
  region: string | null;
  city: string;
  decisionCount: number;
}) {
  return (
    <header>
      <div className="relative bg-bone-deep border border-line rounded-2xl aspect-[16/7] md:aspect-[16/5] overflow-hidden">
        {cover.kind === "image" ? (
          // eslint-disable-next-line @next/next/no-img-element -- data URL preview; backend serves signed S3 URL.
          <img
            src={cover.url}
            alt=""
            className="size-full object-cover"
            draggable={false}
          />
        ) : (
          <div
            aria-hidden
            className="size-full"
            style={{
              background: `linear-gradient(135deg, ${cover.from}, ${cover.to})`,
            }}
          />
        )}
      </div>
      <div className="mt-6 flex flex-col md:flex-row md:items-end md:gap-5 gap-4">
        <AvatarDisc name={displayName} size="lg" tone="ink" avatarOverride={null} />
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            Photographer
          </p>
          <div className="mt-2 flex items-baseline gap-3 flex-wrap">
            <h1 className="font-display text-3xl md:text-4xl font-medium tracking-tight text-ink leading-[1.05]">
              {displayName}
            </h1>
            <StatusPill label={statusLabel} tone={statusTone} />
          </div>
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-3">
            {handle ? `@${handle}` : "No handle yet"}
            <span className="text-slate-soft"> · </span>
            <span className="tnum">since {formatMemberSince(memberSince)}</span>
          </p>
          <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-2">
            {region ?? "Region not set"}
            <span className="text-slate-soft"> · </span>
            {city}
            <span className="text-slate-soft"> · </span>
            <span className="tnum">{decisionCount}</span>{" "}
            actions on file
          </p>
        </div>
      </div>
    </header>
  );
}

function StatusPill({
  label,
  tone,
}: {
  label: string;
  tone: "fresh" | "ink" | "slate";
}) {
  const toneClass =
    tone === "fresh"
      ? "border-fresh/30 text-fresh"
      : tone === "ink"
        ? "border-ink text-ink"
        : "border-line text-slate";
  return (
    <span
      className={`inline-flex items-center font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] rounded-full border px-3 py-0.5 ${toneClass}`}
    >
      {label}
    </span>
  );
}

/* ─────────────── COMPLETENESS ─────────────── */

function CompletenessSlab({
  snapshot,
  liveSettings,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
  liveSettings: ReturnType<typeof useAdminPhotographerView> extends infer V
    ? V extends { liveSettings: infer L }
      ? L
      : never
    : never;
}) {
  // When viewing the session photographer, derive completeness from live
  // store state instead of the seeded snapshot so admin sees the latest.
  const effectiveSnapshot: PhotographerSettingsSnapshot | null = liveSettings
    ? {
        hasCover: !!liveSettings.cover,
        hasBrandName: liveSettings.brandName.trim().length > 0,
        hasWatermark: !!liveSettings.watermark,
        hasHandle: liveSettings.handle.trim().length >= 3,
        hasRegion: !!liveSettings.region,
        socialCount: liveSettings.socials.length,
        payoutCount: liveSettings.payouts.length,
      }
    : snapshot;

  const filledFields = effectiveSnapshot
    ? COMPLETENESS_FIELDS.reduce(
        (acc, f) => acc + (effectiveSnapshot[f.key] ? 1 : 0),
        0,
      )
    : 0;
  const socials = effectiveSnapshot?.socialCount ?? 0;
  const payouts = effectiveSnapshot?.payoutCount ?? 0;
  const filledTotal = filledFields + (socials > 0 ? 1 : 0) + (payouts > 0 ? 1 : 0);
  const total = COMPLETENESS_FIELDS.length + 2;

  return (
    <Slab
      id="completeness"
      number="01"
      title="Completeness"
      caption="Settings checklist"
      trailing={`${filledTotal}/${total} fields`}
    >
      {!effectiveSnapshot ? (
        <p className="font-sans text-sm text-slate-soft">
          No settings on file.
        </p>
      ) : (
        <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {COMPLETENESS_FIELDS.map((field) => (
            <CompletenessRow
              key={field.key}
              label={field.label}
              filled={!!effectiveSnapshot[field.key]}
            />
          ))}
          <CompletenessRow
            label="Social profiles"
            filled={socials > 0}
            detail={`${socials} on file`}
          />
          <CompletenessRow
            label="Payout accounts"
            filled={payouts > 0}
            detail={`${payouts} on file`}
          />
        </ul>
      )}
    </Slab>
  );
}

function CompletenessRow({
  label,
  filled,
  detail,
}: {
  label: string;
  filled: boolean;
  detail?: string;
}) {
  return (
    <li className="flex items-center justify-between gap-3 border-b border-line pb-2.5">
      <div className="flex items-center gap-3 min-w-0">
        <span
          aria-hidden
          className={`size-2 rounded-full shrink-0 ${filled ? "bg-fresh" : "bg-line"}`}
        />
        <p className="font-sans text-sm text-ink-soft truncate">{label}</p>
      </div>
      <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft tnum shrink-0">
        {detail ?? (filled ? "set" : "missing")}
      </p>
    </li>
  );
}

/* ─────────────── ABOUT ─────────────── */

function AboutSlab({
  bio,
  region,
  city,
  memberSince,
  email,
}: {
  bio: string;
  region: string | null;
  city: string;
  memberSince: string;
  email: string;
}) {
  return (
    <Slab id="about" number="02" title="About" caption="Profile basics">
      <p className="font-sans text-base text-ink-soft leading-relaxed max-w-2xl">
        {bio}
      </p>
      <dl className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8 max-w-2xl">
        <DefRow label="Region" value={region ?? "—"} />
        <DefRow label="City" value={city} />
        <DefRow label="Joined" value={formatLongDate(memberSince, true)} />
        <DefRow label="Email" value={email} mono />
      </dl>
    </Slab>
  );
}

function DefRow({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div>
      <dt className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft">
        {label}
      </dt>
      <dd
        className={`mt-1 text-sm text-ink truncate ${mono ? "font-mono" : "font-sans"}`}
      >
        {value}
      </dd>
    </div>
  );
}

/* ─────────────── PAYOUTS ─────────────── */

function PayoutsSlab({
  liveSettings,
}: {
  liveSettings: ReturnType<typeof useAdminPhotographerView> extends infer V
    ? V extends { liveSettings: infer L }
      ? L
      : never
    : never;
}) {
  if (!liveSettings) {
    return (
      <Slab
        id="payouts"
        number="03"
        title="Payouts"
        caption="Sales destinations"
      >
        <p className="font-sans text-sm text-slate-soft">
          Payout details visible only when this photographer is signed in.
        </p>
      </Slab>
    );
  }
  const payouts = liveSettings.payouts;
  return (
    <Slab
      id="payouts"
      number="03"
      title="Payouts"
      caption="Sales destinations"
      trailing={`${payouts.length} ${payouts.length === 1 ? "account" : "accounts"}`}
    >
      {payouts.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No payout accounts on file.
        </p>
      ) : (
        <ul className="space-y-3">
          {payouts.map((p) => (
            <li
              key={p.id}
              className="flex items-center justify-between gap-4 border-b border-line pb-3"
            >
              <div className="min-w-0">
                <p className="font-display text-base text-ink truncate">
                  {PAYOUT_METHOD_LABEL[p.method]}
                  {p.isPrimary && (
                    <span className="ml-2 font-mono uppercase tracking-[0.25em] text-[10px] text-fresh tnum">
                      Primary
                    </span>
                  )}
                </p>
                <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1 tnum">
                  {formatPayoutNumber(p.method, p.accountNumber)}
                </p>
              </div>
              <p className="font-sans text-sm text-slate-soft truncate">
                {p.accountName}
              </p>
            </li>
          ))}
        </ul>
      )}
    </Slab>
  );
}

/* ─────────────── EVENTS COVERED ─────────────── */

function EventsCoveredSlab({
  events,
  catalogResolver,
}: {
  events: ReadonlyArray<{
    eventSlug: string;
    state: string;
    photoCount: number;
    salesCount: number;
  }>;
  catalogResolver: (slug: string) => string | undefined;
}) {
  return (
    <Slab
      id="events"
      number="04"
      title="Events covered"
      caption="Race history"
      trailing={`${events.length} ${events.length === 1 ? "event" : "events"}`}
    >
      {events.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No events on file yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {events.map((coverage) => {
            const eventName =
              catalogResolver(coverage.eventSlug) ?? coverage.eventSlug;
            return (
              <li
                key={coverage.eventSlug}
                className="flex items-center justify-between gap-4 border-b border-line pb-3"
              >
                <div className="min-w-0">
                  <p className="font-display text-base text-ink truncate">
                    {eventName}
                  </p>
                  <p className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate-soft mt-1 tnum">
                    {coverage.state.toUpperCase()}
                    <span className="text-slate-soft"> · </span>
                    <span>{coverage.photoCount.toLocaleString()} photos</span>
                    <span className="text-slate-soft"> · </span>
                    <span>{coverage.salesCount.toLocaleString()} sold</span>
                  </p>
                </div>
              </li>
            );
          })}
        </ul>
      )}
    </Slab>
  );
}

/* ─────────────── ACTIVITY ─────────────── */

function ActivitySlab({ userId }: { userId: string }) {
  return (
    <Slab
      id="activity"
      number="05"
      title="Activity"
      caption="Decisions on file"
    >
      <AdminDecisionsTimeline
        userId={userId}
        limit={20}
        emptyCopy="No actions on file yet."
      />
    </Slab>
  );
}
