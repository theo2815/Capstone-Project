"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { CollapsibleReviewSlab } from "@/components/admin/admin-collapse";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { Kicker } from "@/components/ui/kicker";
import { AdminActionAside } from "@/components/admin/admin-action-aside";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import { AdminStatusPill, type AdminStatusPillTone } from "@/components/admin/admin-status-pill";
import {
  useAdminPhotographerView,
  syntheticCoverGradient,
  computeCompleteness,
  photographerStatus,
  snapshotFromSettings,
} from "@/lib/admin-photographer-view";
import type { EffectivePhotographerSettings } from "@/lib/admin-photographer-view";
import {
  CompletenessSection,
  PayoutsSection,
  PublicUrlSection,
  SocialsSection,
  WatermarkSection,
} from "@/components/admin/photographer-review-sections";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatMemberSince, formatRaceDate, sentenceCaseDate } from "@/lib/format";
import { useAdminUserRecord } from "@/hooks/use-admin-users";
import type { AdminPhotographerRecord } from "@/lib/api-admin";
import { SellerStatusPill } from "@/components/admin/seller-details-queue";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";
import {
  BRAND_COLOR_HEX,
  BRAND_COLOR_LABEL,
} from "@/store/photographer-settings-store";
import type { CoverSource } from "@/lib/photographer-registry";
import type { PhotographerSettingsSnapshot } from "@/lib/admin-user-registry";
import { formatRegionLabel } from "@/lib/ph-regions";
import { useRegions } from "@/hooks/use-regions";

export default function AdminPhotographerDetailPage() {
  const params = useParams<{ handle: string }>();
  const raw = Array.isArray(params.handle) ? params.handle[0] : params.handle;
  const handle = (raw ?? "").trim().toLowerCase();
  const { view, loading, error, refetch } = useAdminPhotographerView(handle);
  const record = useAdminUserRecord(view?.row.userId ?? null);

  if (loading && !view) {
    return (
      <div className="space-y-6" role="status" aria-busy="true">
        <span className="sr-only">Loading photographer…</span>
        <Skeleton className="h-10 w-48" />
        <Skeleton className="h-40 w-full rounded-2xl" />
        <Skeleton className="h-64 w-full rounded-2xl" />
      </div>
    );
  }
  if (error) {
    return (
      <div role="alert" className="space-y-3">
        <p className="font-sans text-sm text-error">Couldn&apos;t load this photographer. Please try again.</p>
        <button type="button" onClick={() => void refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
          Retry
        </button>
      </div>
    );
  }

  if (!view) {
    notFound();
  }

  const { row, effectiveSettings, decisions } = view;
  const displayName =
    effectiveSettings?.brandName?.trim() || row.brandName || row.name;
  const cover: CoverSource =
    effectiveSettings?.cover ??
    { kind: "gradient", ...syntheticCoverGradient(row.userId) };
  const bio = effectiveSettings?.bio?.trim() || "Bio not set yet.";
  const memberSinceIso = row.createdAt;
  const status = photographerStatus(row);
  const publicHandle = effectiveSettings?.handle?.trim() || row.handle || "";

  return (
    <>
      <Link href={ROUTES.ADMIN_PHOTOGRAPHERS} className="inline-flex min-h-[44px] items-center text-sm text-slate hover:text-ink">
        ← Photographers
      </Link>

      <Hero
        cover={cover}
        displayName={displayName}
        handle={publicHandle || null}
        statusLabel={status.label}
        statusTone={status.tone}
        memberSince={memberSinceIso}
        region={row.region}
        city={row.city}
        decisionCount={decisions.length}
        brandColor={effectiveSettings?.brandColor}
        avatarUrl={effectiveSettings?.avatarUrl ?? row.avatarUrl ?? null}
        record={record.data ?? null}
      />

      {/* Actions come first in the DOM, so a phone reaches them before the eight sections; lg moves them to the side. */}
      <div className="mt-10 lg:mt-14 grid lg:grid-cols-[1fr_18rem] lg:gap-12 lg:items-start">
        <AdminActionAside row={row} />
        <div className="min-w-0 lg:col-start-1 lg:row-start-1">
          <CompletenessSlab
            snapshot={row.settingsSnapshot}
            effective={effectiveSettings}
          />
          <AboutSlab
            bio={bio}
            region={row.region}
            regionCodes={effectiveSettings?.region ?? null}
            city={row.city}
            memberSince={memberSinceIso}
            email={row.email}
            brandColor={effectiveSettings?.brandColor}
          />
          <PublicUrlSlab handle={publicHandle} />
          <WatermarkSlab effective={effectiveSettings} brand={displayName} />
          <SocialsSlab effective={effectiveSettings} />
          <PayoutsSlab effective={effectiveSettings} />
          <EventsCoveredSlab record={record.data} failed={record.isError} onRetry={() => void record.refetch()} />
          <ActivitySlab userId={row.userId} />
        </div>
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
  brandColor,
  avatarUrl,
  record,
}: {
  cover: CoverSource;
  displayName: string;
  handle: string | null;
  statusLabel: string;
  statusTone: AdminStatusPillTone;
  memberSince: string;
  region: string | null;
  city: string;
  decisionCount: number;
  brandColor?: EffectivePhotographerSettings["brandColor"];
  avatarUrl: string | null;
  record: AdminPhotographerRecord | null;
}) {
  const brandHex =
    brandColor && brandColor !== "none" ? BRAND_COLOR_HEX[brandColor] : null;
  return (
    <header className="mt-4 pb-8 md:pb-12 border-b border-line">
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
        <AvatarDisc
          name={displayName}
          size="lg"
          tone="ink"
          avatarOverride={avatarUrl ? { dataUrl: avatarUrl } : null}
        />
        <div className="flex-1 min-w-0">
          <Kicker as="p">
            Photographer
          </Kicker>
          <div className="mt-3 flex items-baseline gap-3 flex-wrap">
            <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink leading-[1.05] flex items-baseline gap-2.5 min-w-0 break-words">
              {brandHex && (
                <span
                  aria-hidden
                  className="inline-block size-2 rounded-full translate-y-[-2px] shrink-0"
                  style={{ backgroundColor: brandHex }}
                />
              )}
              {displayName}
            </h1>
            <AdminStatusPill label={statusLabel} tone={statusTone} />
          </div>
          <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl break-words">
            {handle ? `@${handle}` : "No handle yet"} · member since{" "}
            <span className="tnum">{sentenceCaseDate(formatMemberSince(memberSince))}</span> ·{" "}
            {region ?? "Region not set"} · {city} · <span className="tnum">{decisionCount}</span> actions on file
          </p>
        </div>
      </div>
      {/* PD8: what an admin weighs before acting; hidden until the record loads (an older backend has none). */}
      {record && (
        <dl className="mt-6 flex flex-wrap gap-x-8 gap-y-4">
          <div>
            <dt className="text-sm text-slate">Seller details</dt>
            <dd className="mt-1">
              {record.sellerDetailsStatus
                ? <SellerStatusPill reviewState={record.sellerDetailsStatus} />
                : <span className="text-sm text-ink">Not submitted</span>}
            </dd>
          </div>
          <div>
            <dt className="text-sm text-slate">Sales</dt>
            <dd className="mt-1 font-display text-xl font-bold text-ink tnum">{record.salesCount.toLocaleString()}</dd>
          </div>
          <div>
            <dt className="text-sm text-slate">Photos</dt>
            <dd className="mt-1 font-display text-xl font-bold text-ink tnum">{record.photoCount.toLocaleString()}</dd>
          </div>
        </dl>
      )}
    </header>
  );
}

/* ─────────────── COMPLETENESS ─────────────── */

function CompletenessSlab({
  snapshot,
  effective,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
  effective: EffectivePhotographerSettings | null;
}) {
  // Live settings win over the directory snapshot, so admin sees what is actually filled in.
  const effectiveSnapshot = effective ? snapshotFromSettings(effective) : snapshot;
  const { filled, total } = computeCompleteness(effectiveSnapshot);

  return (
    <CollapsibleReviewSlab
      id="completeness"
      number="01"
      title="Completeness"
      caption="Settings checklist"
      trailing={`${filled}/${total} fields`}
    >
      <CompletenessSection snapshot={effectiveSnapshot} />
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── ABOUT ─────────────── */

function AboutSlab({
  bio,
  region,
  regionCodes,
  city,
  memberSince,
  email,
  brandColor,
}: {
  bio: string;
  region: string | null;
  regionCodes: { regionCode: string; provinceCode: string } | null;
  city: string;
  memberSince: string;
  email: string;
  brandColor?: EffectivePhotographerSettings["brandColor"];
}) {
  const { regions } = useRegions();
  const resolvedRegion = regionCodes
    ? formatRegionLabel(regions, regionCodes.regionCode, regionCodes.provinceCode)
    : null;
  const regionLine = resolvedRegion ?? region ?? "—";
  return (
    <CollapsibleReviewSlab id="about" number="02" title="About" caption="Profile basics">
      <p className="font-sans text-base text-ink-soft leading-relaxed max-w-2xl">
        {bio}
      </p>
      <dl className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-y-4 gap-x-8 max-w-2xl">
        <DefRow label="Region" value={regionLine} />
        <DefRow label="City" value={city} />
        <DefRow label="Joined" value={sentenceCaseDate(formatLongDate(memberSince, true))} />
        <DefRow label="Email" value={email} mono />
        {brandColor && brandColor !== "none" && (
          <DefRow label="Brand accent" value={BRAND_COLOR_LABEL[brandColor]} />
        )}
      </dl>
    </CollapsibleReviewSlab>
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
      <Kicker as="dt" tone="soft">
        {label}
      </Kicker>
      <dd
        title={value}
        className={`mt-1 text-sm text-ink break-all ${mono ? "font-mono" : "font-sans"}`}
      >
        {value}
      </dd>
    </div>
  );
}

/* ─────────────── PUBLIC URL ─────────────── */

function PublicUrlSlab({ handle }: { handle: string }) {
  return (
    <CollapsibleReviewSlab
      id="public-url"
      number="03"
      title="Public URL"
      caption="Where runners find this photographer"
      trailing={handle ? `@${handle}` : undefined}
    >
      <PublicUrlSection handle={handle} />
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── WATERMARK ─────────────── */

function WatermarkSlab({
  effective,
  brand,
}: {
  effective: EffectivePhotographerSettings | null;
  brand: string;
}) {
  return (
    <CollapsibleReviewSlab
      id="watermark"
      number="04"
      title="Watermark"
      caption="Stamped on every preview"
      trailing={effective?.watermark ? "On file" : "Not set"}
    >
      <WatermarkSection effective={effective} brand={brand} />
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── SOCIALS ─────────────── */

function SocialsSlab({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const count = effective?.socials.length ?? 0;
  return (
    <CollapsibleReviewSlab
      id="socials"
      number="05"
      title="Social & verification links"
      caption="Public proof of identity"
      trailing={`${count} ${count === 1 ? "link" : "links"}`}
    >
      <SocialsSection effective={effective} />
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── PAYOUTS ─────────────── */

function PayoutsSlab({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const count = effective?.payouts.length ?? 0;
  return (
    <CollapsibleReviewSlab
      id="payouts"
      number="06"
      title="Payout accounts"
      caption="Sales destinations"
      trailing={`${count} ${count === 1 ? "account" : "accounts"}`}
    >
      <PayoutsSection effective={effective} />
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── EVENTS COVERED ─────────────── */

function EventsCoveredSlab({
  record,
  failed,
  onRetry,
}: {
  /** Undefined while loading; null when the backend has no record route yet. */
  record: AdminPhotographerRecord | null | undefined;
  failed: boolean;
  onRetry: () => void;
}) {
  const total = record?.eventsTotal ?? 0;
  return (
    <CollapsibleReviewSlab
      id="events"
      number="07"
      title="Events covered"
      caption="Race history"
      trailing={record ? `${total.toLocaleString()} ${total === 1 ? "event" : "events"}` : undefined}
    >
      {failed && !record ? (
        <div role="alert" className="space-y-3">
          <p className="font-sans text-sm text-error">Couldn&apos;t load the events. Please try again.</p>
          <button type="button" onClick={onRetry} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>Retry</button>
        </div>
      ) : record === undefined ? (
        <div className="space-y-3" aria-busy="true">
          <Skeleton className="h-12 w-full" />
          <Skeleton className="h-12 w-full" />
        </div>
      ) : record === null ? (
        <p className="font-sans text-sm text-slate-soft">Event history isn&apos;t available yet.</p>
      ) : record.events.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">No events on file yet.</p>
      ) : (
        <>
          {total > record.events.length && (
            <p className="font-sans text-sm text-slate mb-4 tnum">
              Showing {record.events.length} of {total.toLocaleString()}, newest first.
            </p>
          )}
          <ul className="space-y-3">
            {record.events.map((event) => (
              <li key={event.eventId} className="border-b border-line pb-3 min-w-0">
                <p className="font-display text-base text-ink break-words">{event.name}</p>
                <p className="font-sans text-sm text-slate-soft mt-1 tnum">
                  {sentenceCaseDate(formatRaceDate(event.date))} · {event.visibility === "unlisted" ? "Private" : "Public"} ·{" "}
                  {event.photoCount.toLocaleString()} photos
                </p>
              </li>
            ))}
          </ul>
        </>
      )}
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── ACTIVITY ─────────────── */

function ActivitySlab({ userId }: { userId: string }) {
  return (
    <CollapsibleReviewSlab
      id="activity"
      number="08"
      title="Activity"
      caption="Decisions on file"
    >
      <AdminDecisionsTimeline
        userId={userId}
        limit={20}
        emptyCopy="No actions on file yet."
      />
    </CollapsibleReviewSlab>
  );
}
