import Link from "next/link";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { completenessItems, type EffectivePhotographerSettings } from "@/lib/admin-photographer-view";
import type { PhotographerSettingsSnapshot } from "@/lib/admin-user-registry";
import { formatPayoutNumber } from "@/lib/payout-format";
import { safeHttpUrl } from "@/lib/utils";
import {
  PAYOUT_METHOD_LABEL,
  SOCIAL_PLATFORM_LABEL,
  SOCIAL_PLATFORM_TILE,
} from "@/store/photographer-settings-store";

// The photographer review sections, shared by the verification drawer and the
// photographer page. Each caller keeps its own wrapper (drawer section or page slab).

const MONO_LABEL = `font-mono uppercase ${KICKER_SIZE_CLASS.sm}`;

export function CoverSection({ effective }: { effective: EffectivePhotographerSettings | null }) {
  const cover = effective?.cover ?? null;
  if (!cover) {
    return <p className="font-sans text-sm text-slate-soft">No cover banner uploaded.</p>;
  }
  return (
    <div className="relative bg-bone-deep border border-line rounded-2xl aspect-[16/7] md:aspect-[16/5] overflow-hidden">
      {cover.kind === "image" ? (
        // eslint-disable-next-line @next/next/no-img-element -- data URL preview; backend serves signed S3 URL.
        <img src={cover.url} alt="" className="size-full object-cover" draggable={false} />
      ) : (
        <div
          aria-hidden
          className="size-full"
          style={{ background: `linear-gradient(135deg, ${cover.from}, ${cover.to})` }}
        />
      )}
    </div>
  );
}

export function PublicUrlSection({ handle }: { handle: string }) {
  if (!handle) {
    return <p className="font-sans text-sm text-slate-soft">No handle set yet.</p>;
  }
  return (
    <div className="flex items-center justify-between gap-4 flex-wrap rounded-2xl border border-line bg-bone-deep px-4 py-3">
      <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-ink tnum truncate">
        quickpitik.com/{handle}
      </p>
      <Link
        href={`/${handle}`}
        target="_blank"
        rel="noopener"
        className={`shrink-0 inline-flex min-h-[44px] items-center ${MONO_LABEL} text-slate hover:text-ink transition-colors`}
      >
        Open ↗
      </Link>
    </div>
  );
}

export function WatermarkSection({
  effective,
  brand,
}: {
  effective: EffectivePhotographerSettings | null;
  brand: string;
}) {
  const watermark = effective?.watermark ?? null;
  if (!watermark) {
    return <p className="font-sans text-sm text-slate-soft">No watermark uploaded.</p>;
  }
  return (
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
  );
}

export function SocialsSection({ effective }: { effective: EffectivePhotographerSettings | null }) {
  const socials = effective?.socials ?? [];
  if (socials.length === 0) {
    return <p className="font-sans text-sm text-slate-soft">No social profiles linked yet.</p>;
  }
  return (
    <ul className="space-y-3">
      {socials.map((social) => {
        const href = safeHttpUrl(social.url);
        return (
          <li key={social.id} className="flex items-center gap-4 border-b border-line pb-3">
            <span
              aria-hidden
              className="shrink-0 size-10 rounded-xl border border-line bg-bone-deep flex items-center justify-center font-mono uppercase tracking-[0.15em] text-[13px] text-ink"
            >
              {SOCIAL_PLATFORM_TILE[social.platform]}
            </span>
            <div className="min-w-0 flex-1">
              <p className="font-display text-base text-ink">{SOCIAL_PLATFORM_LABEL[social.platform]}</p>
              <p className="font-mono text-[13px] min-[400px]:text-[14px] md:text-[13px] text-slate-soft mt-0.5 truncate">
                {social.url}
              </p>
            </div>
            {href && (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className={`shrink-0 inline-flex min-h-[44px] items-center ${MONO_LABEL} text-slate hover:text-ink transition-colors`}
              >
                Open ↗
              </a>
            )}
          </li>
        );
      })}
    </ul>
  );
}

export function PayoutsSection({ effective }: { effective: EffectivePhotographerSettings | null }) {
  const payouts = effective?.payouts ?? [];
  if (payouts.length === 0) {
    return <p className="font-sans text-sm text-slate-soft">No payout accounts on file.</p>;
  }
  return (
    <ul className="space-y-3">
      {payouts.map((p) => (
        <li key={p.id} className="flex items-center justify-between gap-4 border-b border-line pb-3">
          <div className="min-w-0">
            <p className="font-display text-base text-ink truncate">
              {PAYOUT_METHOD_LABEL[p.method]}
              {p.isPrimary && (
                <Kicker tone="active" tnum className="ml-2">
                  Primary
                </Kicker>
              )}
            </p>
            <p className={`${MONO_LABEL} text-slate-soft mt-1 tnum`}>
              {formatPayoutNumber(p.method, p.accountNumber)}
            </p>
          </div>
          <p className="font-sans text-sm text-slate-soft truncate">{p.accountName}</p>
        </li>
      ))}
    </ul>
  );
}

export function CompletenessSection({ snapshot }: { snapshot: PhotographerSettingsSnapshot | null }) {
  if (!snapshot) {
    return <p className="font-sans text-sm text-slate-soft">No settings on file.</p>;
  }
  return (
    <ul className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {completenessItems(snapshot).map((item) => (
        <li key={item.label} className="flex items-center justify-between gap-3 border-b border-line pb-2.5">
          <div className="flex items-center gap-3 min-w-0">
            <span aria-hidden className={`size-2 rounded-full shrink-0 ${item.filled ? "bg-fresh" : "bg-line"}`} />
            <p className="font-sans text-sm text-ink-soft truncate">{item.label}</p>
          </div>
          <p className={`${MONO_LABEL} text-slate-soft tnum shrink-0`}>
            {item.detail ?? (item.filled ? "set" : "missing")}
          </p>
        </li>
      ))}
    </ul>
  );
}
