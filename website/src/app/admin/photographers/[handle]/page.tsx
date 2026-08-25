"use client";

import Link from "next/link";
import { notFound, useParams } from "next/navigation";
import { CollapsibleReviewSlab } from "@/components/admin/admin-collapse";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { Kicker } from "@/components/ui/kicker";
import { AdminActionAside } from "@/components/admin/admin-action-aside";
import { AdminDecisionsTimeline } from "@/components/admin/admin-decisions-timeline";
import {
  useAdminPhotographerView,
  syntheticCoverGradient,
} from "@/lib/admin-photographer-view";
import type { EffectivePhotographerSettings } from "@/lib/admin-photographer-view";
import { ROUTES } from "@/lib/constants";
import { formatLongDate, formatMemberSince } from "@/lib/format";
import { useEventCatalog } from "@/lib/event-catalog";
import {
  BRAND_COLOR_HEX,
  BRAND_COLOR_LABEL,
  PAYOUT_METHOD_LABEL,
  SOCIAL_PLATFORM_LABEL,
  SOCIAL_PLATFORM_TILE,
} from "@/store/photographer-settings-store";
import type { CoverSource } from "@/lib/photographer-registry";
import type { PhotographerSettingsSnapshot } from "@/lib/admin-user-registry";
import { formatPayoutNumber } from "@/lib/payout-format";
import { formatRegionLabel } from "@/lib/ph-regions";
import { useRegions } from "@/hooks/use-regions";

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

const PUBLIC_URL_PREFIX = "quickpitik.com";

// Phase 2a admin photographer detail. Hero + 8 content slabs (completeness,
// about, public URL, watermark, socials, payouts, events covered, activity)
// + sticky-on-lg <AdminActionAside>. Reads via useAdminPhotographerView()
// so admin store overrides + live settings flow through to the visible
// state without a refresh. Non-session photographers (anyone other than
// the logged-in user) read full settings from the admin seed module so
// admin can verify cover/watermark/region/socials/payouts in one place.
export default function AdminPhotographerDetailPage() {
  const params = useParams<{ handle: string }>();
  const raw = Array.isArray(params.handle) ? params.handle[0] : params.handle;
  const handle = (raw ?? "").trim().toLowerCase();
  const view = useAdminPhotographerView(handle);
  const catalog = useEventCatalog();

  if (!view) {
    notFound();
  }

  const { row, profile, effectiveSettings, decisions } = view;
  const displayName =
    effectiveSettings?.brandName?.trim() || row.brandName || row.name;
  const cover: CoverSource =
    effectiveSettings?.cover ??
    profile?.cover ??
    { kind: "gradient", ...syntheticCoverGradient(row.userId) };
  const bio =
    effectiveSettings?.bio?.trim() || profile?.bio || "Bio not set yet.";
  const memberSinceIso = profile?.memberSince ?? row.createdAt;
  const isSuspended = row.suspendedAt !== null;
  const statusLabel = isSuspended
    ? "Suspended"
    : STATUS_LABEL[row.verificationStatus];
  const statusTone: "fresh" | "ink" | "slate" = isSuspended
    ? "ink"
    : row.verificationStatus === "approved"
      ? "fresh"
      : "slate";
  const publicHandle = effectiveSettings?.handle?.trim() || row.handle || "";

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
        handle={publicHandle || null}
        statusLabel={statusLabel}
        statusTone={statusTone}
        memberSince={memberSinceIso}
        region={row.region}
        city={row.city}
        decisionCount={decisions.length}
        brandColor={effectiveSettings?.brandColor}
        avatarUrl={effectiveSettings?.avatarUrl ?? row.avatarUrl ?? null}
      />

      <div className="mt-10 lg:mt-14 grid lg:grid-cols-[1fr_18rem] lg:gap-12 lg:items-start">
        <div>
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
  brandColor,
  avatarUrl,
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
  brandColor?: EffectivePhotographerSettings["brandColor"];
  avatarUrl: string | null;
}) {
  const brandHex =
    brandColor && brandColor !== "none" ? BRAND_COLOR_HEX[brandColor] : null;
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
          <div className="mt-2 flex items-baseline gap-3 flex-wrap">
            <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight text-ink leading-[1.05] flex items-baseline gap-2.5">
              {brandHex && (
                <span
                  aria-hidden
                  className="inline-block size-2 rounded-full translate-y-[-2px] shrink-0"
                  style={{ backgroundColor: brandHex }}
                />
              )}
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
  effective,
}: {
  snapshot: PhotographerSettingsSnapshot | null;
  effective: EffectivePhotographerSettings | null;
}) {
  // When effective settings are present (live OR seed), derive completeness
  // from them so admin sees what's actually filled in for this photographer
  // — not just the directory snapshot.
  const effectiveSnapshot: PhotographerSettingsSnapshot | null = effective
    ? {
        hasCover: !!effective.cover,
        hasBrandName: effective.brandName.trim().length > 0,
        hasWatermark: !!effective.watermark,
        hasHandle: effective.handle.trim().length >= 3,
        hasRegion: !!effective.region,
        socialCount: effective.socials.length,
        payoutCount: effective.payouts.length,
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
    <CollapsibleReviewSlab
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
    </CollapsibleReviewSlab>
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
        <DefRow label="Joined" value={formatLongDate(memberSince, true)} />
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

/* ─────────────── PUBLIC URL ─────────────── */

function PublicUrlSlab({ handle }: { handle: string }) {
  if (!handle) {
    return (
      <CollapsibleReviewSlab
        id="public-url"
        number="03"
        title="Public URL"
        caption="Where runners find this photographer"
      >
        <p className="font-sans text-sm text-slate-soft">
          No handle set yet.
        </p>
      </CollapsibleReviewSlab>
    );
  }
  const display = `${PUBLIC_URL_PREFIX}/${handle}`;
  return (
    <CollapsibleReviewSlab
      id="public-url"
      number="03"
      title="Public URL"
      caption="Where runners find this photographer"
      trailing={`@${handle}`}
    >
      <div className="flex items-center justify-between gap-4 flex-wrap rounded-2xl border border-line bg-bone-deep px-4 py-3">
        <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-ink tnum truncate">
          {display}
        </p>
        <Link
          href={`/${handle}`}
          target="_blank"
          rel="noopener"
          className="shrink-0 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
        >
          Open ↗
        </Link>
      </div>
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
  const watermark = effective?.watermark ?? null;
  return (
    <CollapsibleReviewSlab
      id="watermark"
      number="04"
      title="Watermark"
      caption="Stamped on every preview"
      trailing={watermark ? "On file" : "Not set"}
    >
      {!watermark ? (
        <p className="font-sans text-sm text-slate-soft">
          Photographer hasn&apos;t uploaded a watermark yet.
        </p>
      ) : (
        <div className="rounded-2xl border border-line bg-bone-deep p-6 md:p-8 flex items-center justify-center min-h-[120px]">
          {watermark.kind === "image" ? (
            // eslint-disable-next-line @next/next/no-img-element -- data URL preview from photographer settings store; backend serves signed S3 URL.
            <img
              src={watermark.dataUrl}
              alt={`${brand} watermark`}
              className="max-h-24 max-w-full object-contain"
              draggable={false}
            />
          ) : (
            <span
              className="font-mono uppercase tracking-[0.4em] text-2xl md:text-3xl text-ink/70"
              aria-label={`Watermark label: ${watermark.label}`}
            >
              © {watermark.label}
            </span>
          )}
        </div>
      )}
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── SOCIALS ─────────────── */

function SocialsSlab({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const socials = effective?.socials ?? [];
  return (
    <CollapsibleReviewSlab
      id="socials"
      number="05"
      title="Social & verification links"
      caption="Public proof of identity"
      trailing={`${socials.length} ${socials.length === 1 ? "link" : "links"}`}
    >
      {socials.length === 0 ? (
        <p className="font-sans text-sm text-slate-soft">
          No social profiles linked yet.
        </p>
      ) : (
        <ul className="space-y-3">
          {socials.map((social) => (
            <li
              key={social.id}
              className="flex items-center gap-4 border-b border-line pb-3"
            >
              <span
                aria-hidden
                className="shrink-0 size-10 rounded-xl border border-line bg-bone-deep flex items-center justify-center font-mono uppercase tracking-[0.15em] text-[13px] text-ink"
              >
                {SOCIAL_PLATFORM_TILE[social.platform]}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-display text-base text-ink">
                  {SOCIAL_PLATFORM_LABEL[social.platform]}
                </p>
                <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-slate-soft mt-0.5 truncate">
                  {social.url}
                </p>
              </div>
              <a
                href={social.url}
                target="_blank"
                rel="noopener noreferrer"
                className="shrink-0 font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
              >
                Open ↗
              </a>
            </li>
          ))}
        </ul>
      )}
    </CollapsibleReviewSlab>
  );
}

/* ─────────────── PAYOUTS ─────────────── */

function PayoutsSlab({
  effective,
}: {
  effective: EffectivePhotographerSettings | null;
}) {
  const payouts = effective?.payouts ?? [];
  return (
    <CollapsibleReviewSlab
      id="payouts"
      number="06"
      title="Payout accounts"
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
                    <Kicker tone="active" tnum className="ml-2">
                      Primary
                    </Kicker>
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
    </CollapsibleReviewSlab>
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
    <CollapsibleReviewSlab
      id="events"
      number="07"
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
