"use client";

import Link from "next/link";
import {
  useCallback,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";
import { Slab } from "@/components/profile-shell";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { Dropdown, DropdownItem } from "@/components/ui/dropdown";
import { Kicker } from "@/components/ui/kicker";
import { useAuth } from "@/hooks/use-auth";
import { useToast } from "@/hooks/use-toast";
import { useUserMediaStore } from "@/store/user-media-store";
import {
  ACCEPTED_IMAGE_MIME,
  fitToDataUrl,
  fitToPngDataUrl,
  squareCropToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";
import {
  PH_REGIONS,
  REGION_GROUP_LABEL,
  formatRegionLabel,
  getRegion,
  type RegionGroup,
} from "@/lib/ph-regions";
import { formatPayoutNumber } from "@/lib/payout-format";
import { validateHandle } from "@/lib/reserved-handles";
import {
  usePhotographerSettingsStore,
  BRAND_COLOR_HEX,
  BRAND_COLOR_LABEL,
  PAYOUT_METHODS,
  PAYOUT_METHOD_HEX,
  PAYOUT_METHOD_LABEL,
  SOCIAL_PLATFORMS,
  SOCIAL_PLATFORM_LABEL,
  SOCIAL_PLATFORM_TILE,
  type BrandColor,
  type PayoutAccount,
  type PayoutMethod,
  type SocialLink,
  type SocialPlatform,
} from "@/store/photographer-settings-store";
import { cn } from "@/lib/utils";

const COVER_MAX_PX = 1920;
const WATERMARK_MAX_PX = 600;
const QR_MAX_PX = 800;
const AVATAR_SIZE_PX = 512;
const BIO_MAX = 140;
const BRAND_NAME_MAX = 50;
const ACCOUNT_NAME_MAX = 64;

const COLOR_ORDER: ReadonlyArray<BrandColor> = [
  "none",
  "fresh",
  "amber",
  "indigo",
  "rose",
  "ink",
];

const REGION_GROUP_ORDER: ReadonlyArray<RegionGroup> = [
  "luzon",
  "visayas",
  "mindanao",
];

export default function SettingsPage() {
  return (
    <>
      <VerificationStatusPanel />
      <PublicProfileSlab />
      <WatermarkSlab />
      <HandleSlab />
      <RegionSlab />
      <SocialSlab />
      <PayoutSlab />
    </>
  );
}

function VerificationStatusPanel() {
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const setStatus = usePhotographerSettingsStore(
    (s) => s.setVerificationStatus,
  );
  const isComplete = usePhotographerSettingsStore((s) => s.isComplete);
  const { showToast } = useToast();
  const complete = isComplete();

  function handleSubmit() {
    setStatus("pending");
    showToast({
      kind: "info",
      message: "Submitted for review. We'll let you know.",
    });
    // TODO(backend): replace this auto-approve simulation with
    // `POST /me/photographer/verification` returning a real review state.
    setTimeout(() => {
      setStatus("approved");
      showToast({
        kind: "success",
        message: "Verified. You can upload to your events now.",
      });
    }, 3000);
  }

  if (status === "approved") {
    return (
      <div className="border border-line rounded-2xl px-5 py-4 bg-bone-deep/30 flex items-center gap-3 mb-8">
        <span
          aria-hidden="true"
          className="size-1.5 rounded-full bg-fresh shrink-0"
        />
        <Kicker as="p">
          Verified · uploads enabled
        </Kicker>
      </div>
    );
  }

  if (status === "pending") {
    return (
      <div className="border border-line rounded-2xl px-5 py-4 bg-bone-deep/40 mb-8">
        <Kicker as="p">
          Awaiting review
        </Kicker>
        <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
          We&apos;re looking over your settings.
        </p>
        <p className="font-sans text-sm text-slate mt-2 max-w-md">
          Reviews take 1–2 business days. You&apos;ll be able to upload as soon
          as we&apos;re done.
        </p>
      </div>
    );
  }

  return (
    <div className="border border-line rounded-2xl px-5 py-5 bg-bone-deep/40 mb-8">
      <Kicker as="p">
        Not yet verified
      </Kicker>
      <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2">
        Fill in every section, then submit for review.
      </p>
      <p className="font-sans text-sm text-slate mt-2 max-w-md">
        Cover, brand, watermark, public URL, region, at least one social link,
        and at least one payout account are required so we can verify you and
        send your sales. Profile picture is optional.
      </p>
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!complete}
        className="mt-5 inline-flex items-center gap-2 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {complete ? "Submit for review" : "Fill the required fields"}
        {complete && <span aria-hidden="true">→</span>}
      </button>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 01 — Public profile (merged: profile picture · cover banner · brand)
// ─────────────────────────────────────────────────────────────────────────
//
// Three sub-sections share one slab so the live preview at the top stays in
// view as the photographer edits — every change in the right column updates
// the left preview immediately. xl+ pins the preview as a sticky column;
// below xl, preview stacks above the controls.

function PublicProfileSlab() {
  // Brand drafts live up here so the preview can read in-flight typing for
  // display name + bio without committing to the persisted store. Cover,
  // accent color, and avatar already commit on click/upload, so the preview
  // reads those straight from the store.
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const bio = usePhotographerSettingsStore((s) => s.bio);
  const [nameDraft, setNameDraft] = useState(brandName);
  const [bioDraft, setBioDraft] = useState(bio);

  return (
    <Slab
      id="public-profile"
      number="01"
      title="Public profile"
      caption="How runners see you on quickpitik.com"
    >
      <div className="xl:grid xl:grid-cols-[minmax(0,360px)_1fr] xl:gap-12">
        <div className="max-w-md xl:max-w-none xl:sticky xl:top-24 xl:self-start">
          <LivePreview nameDraft={nameDraft} bioDraft={bioDraft} />
        </div>
        <div className="mt-10 xl:mt-0">
          <PictureSubsection />
          <div className="border-t border-line/60 mt-12 pt-12">
            <CoverSubsection />
          </div>
          <div className="border-t border-line/60 mt-12 pt-12">
            <BrandSubsection
              nameDraft={nameDraft}
              setNameDraft={setNameDraft}
              bioDraft={bioDraft}
              setBioDraft={setBioDraft}
            />
          </div>
        </div>
      </div>
    </Slab>
  );
}

function SubHeading({ label, caption }: { label: string; caption?: string }) {
  return (
    <div className="flex items-baseline gap-3 flex-wrap mb-6">
      <Kicker as="p">
        {label}
      </Kicker>
      {caption && (
        <Kicker as="p" tone="soft" className="hidden md:block">
          {caption}
        </Kicker>
      )}
    </div>
  );
}

interface LivePreviewProps {
  /** In-flight display-name draft from BrandSubsection (preview is informational; nothing is persisted until Save). */
  nameDraft: string;
  /** In-flight bio draft from BrandSubsection. */
  bioDraft: string;
}

function LivePreview({ nameDraft, bioDraft }: LivePreviewProps) {
  const { user } = useAuth();
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const brandColor = usePhotographerSettingsStore((s) => s.brandColor);
  const handle = usePhotographerSettingsStore((s) => s.handle);

  const accent =
    brandColor !== "none" ? BRAND_COLOR_HEX[brandColor] : null;
  const accentLabel =
    brandColor !== "none" ? BRAND_COLOR_LABEL[brandColor] : null;
  const trimmedHandle = handle.trim();
  const handleValid =
    trimmedHandle.length > 0 && validateHandle(trimmedHandle) === null;
  const trimmedName = nameDraft.trim();
  const displayName = trimmedName || "Photographer";
  // AvatarDisc reads from useUserMediaStore directly (no override) so the
  // preview reflects the live profile picture as soon as it's uploaded.
  const avatarName = trimmedName || user?.name || "Photographer";

  return (
    <div className="border border-line rounded-2xl overflow-hidden bg-bone shadow-sm">
      <div className="px-5 pt-4 pb-3 flex items-center justify-between gap-3 border-b border-line/60">
        <Kicker as="p" tone="soft">
          Live preview
        </Kicker>
        {handleValid ? (
          <Kicker
            as={Link}
            href={`/${trimmedHandle}`}
            target="_blank"
            rel="noopener noreferrer"
            className="hover:text-ink transition-colors inline-flex items-center gap-1"
          >
            Open live <span aria-hidden="true">↗</span>
          </Kicker>
        ) : (
          <Kicker tone="soft">
            Set URL to open
          </Kicker>
        )}
      </div>

      <div className="relative bg-bone-deep aspect-[16/5] overflow-hidden">
        {cover ? (
          // Responsive fit: mobile fills the cover edge-to-edge (object-cover,
          // the photo dominates the small viewport), desktop+ contains the
          // full image (object-contain, bone-deep bands fill any empty space
          // — Facebook-style). The container's bg-bone-deep IS the band fill.
          // eslint-disable-next-line @next/next/no-img-element -- data-URL mock; backend will return signed S3 URLs.
          <img
            src={cover.dataUrl}
            alt=""
            className="size-full object-cover md:object-contain"
            draggable={false}
          />
        ) : (
          <div
            aria-hidden="true"
            className="size-full"
            style={{
              background:
                "linear-gradient(135deg, var(--bone-deep), var(--bone))",
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

      <div className="relative px-5 pb-5 -mt-7 md:-mt-8">
        <AvatarDisc name={avatarName} size="md" />
        <Kicker as="p" tnum className="mt-4 flex items-center gap-2 flex-wrap">
          <span>Photographer</span>
          <span className="text-slate-soft">·</span>
          <span>Cebu</span>
          {accentLabel && (
            <>
              <span className="text-slate-soft">·</span>
              <span className="text-slate-soft">{accentLabel}</span>
            </>
          )}
        </Kicker>

        <div className="mt-3 flex items-baseline gap-2 flex-wrap">
          <h3
            className={cn(
              "font-display text-2xl md:text-3xl font-medium tracking-tight leading-[1.05] break-words min-w-0",
              trimmedName ? "text-ink" : "text-slate-soft",
            )}
          >
            {displayName}
          </h3>
          {accent && (
            <span
              aria-hidden="true"
              className="size-2.5 rounded-full inline-block shrink-0"
              style={{ backgroundColor: accent }}
            />
          )}
        </div>

        {bioDraft.trim() && (
          <p className="font-sans text-sm text-ink-soft mt-2 leading-relaxed line-clamp-3">
            {bioDraft}
          </p>
        )}

        <div className="mt-4">
          <Kicker className="inline-flex items-center gap-2 max-w-full rounded-full border border-line bg-bone-deep/40 px-3 py-1.5 text-ink">
            <span className="text-slate-soft shrink-0">URL</span>
            <span className="text-slate-soft shrink-0" aria-hidden="true">
              ·
            </span>
            <span className="truncate">
              quickpitik.com/{trimmedHandle || "your-handle"}
            </span>
          </Kicker>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Sub-section 01a — Profile picture
// ─────────────────────────────────────────────────────────────────────────

function PictureSubsection() {
  const { user } = useAuth();
  const avatar = useUserMediaStore((s) => s.avatar);
  const setAvatar = useUserMediaStore((s) => s.setAvatar);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setBusy(true);
    setError(null);
    try {
      // TODO(backend): swap for `api.post("/me/avatar", formData)` — server
      // does the same center-crop + JPEG re-encode this helper does.
      const dataUrl = await squareCropToDataUrl(file, AVATAR_SIZE_PX);
      setAvatar({ dataUrl, uploadedAt: new Date().toISOString() });
      showToast({ kind: "success", message: "Profile picture updated." });
    } catch {
      setError("Could not process this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  function handleRemove() {
    setAvatar(null);
    setError(null);
    showToast({ kind: "success", message: "Profile picture removed." });
  }

  if (!user) return null;

  return (
    <div>
      <SubHeading label="Profile picture" caption="Square crop · 512×512" />
      <div className="flex flex-wrap items-center gap-x-6 gap-y-4">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3">
          <label
            className={cn(
              "font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors inline-flex items-center gap-2",
              busy ? "opacity-60 cursor-wait" : "cursor-pointer",
            )}
          >
            {busy
              ? "Processing…"
              : avatar
                ? "Replace picture"
                : "Upload picture"}
            {!busy && <span aria-hidden="true">→</span>}
            <input
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handlePick}
              disabled={busy}
              className="sr-only"
            />
          </label>
          {avatar && !busy && (
            <button
              type="button"
              onClick={handleRemove}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
            >
              Remove
            </button>
          )}
        </div>
      </div>
      {error && (
        <p role="alert" className="font-sans text-sm text-error mt-3">
          {error}
        </p>
      )}
      <p className="font-sans text-sm text-slate-soft mt-4 max-w-md">
        Shown next to your name across QuickPitik. Different from your
        face-search selfies — those live in your{" "}
        <Link
          href="/profile#selfies"
          className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
        >
          selfie library
        </Link>
        .
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Sub-section 01b — Cover banner
// ─────────────────────────────────────────────────────────────────────────

function CoverSubsection() {
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const setCover = usePhotographerSettingsStore((s) => s.setCover);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setBusy(true);
    setError(null);
    try {
      // TODO(backend): swap for `api.post("/me/photographer/cover", formData)`
      // — server-side will run the same downscale + JPEG re-encode.
      const dataUrl = await fitToDataUrl(file, COVER_MAX_PX, 0.82);
      setCover({ dataUrl, uploadedAt: new Date().toISOString() });
      showToast({ kind: "success", message: "Cover updated." });
    } catch {
      setError("Could not process this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  function handleRemove() {
    setCover(null);
    setError(null);
    showToast({ kind: "success", message: "Cover removed." });
  }

  return (
    <div>
      <SubHeading label="Cover banner" caption="16:5 · 1920px wide · 8 MB" />
      {error && (
        <p role="alert" className="font-sans text-sm text-error mb-3">
          {error}
        </p>
      )}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
        <label
          className={cn(
            "font-sans text-sm font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-2.5 px-5 rounded-full transition-colors inline-flex items-center gap-2",
            busy ? "opacity-60 cursor-wait" : "cursor-pointer",
          )}
        >
          {busy ? "Processing…" : cover ? "Replace cover" : "Upload cover"}
          {!busy && <span aria-hidden="true">→</span>}
          <input
            type="file"
            accept={ACCEPTED_IMAGE_MIME.join(",")}
            onChange={handlePick}
            disabled={busy}
            className="sr-only"
          />
        </label>
        {cover && !busy && (
          <button
            type="button"
            onClick={handleRemove}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
          >
            Remove
          </button>
        )}
      </div>
      <p className="font-sans text-sm text-slate-soft mt-4 max-w-md">
        Wide horizontal image — landscape, course, or finish-line shots work
        best. JPEG, PNG, or WebP. Skip portrait photos; they crop badly at
        this aspect.
      </p>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Sub-section 01c — Brand
// ─────────────────────────────────────────────────────────────────────────

interface BrandSubsectionProps {
  nameDraft: string;
  setNameDraft: (value: string) => void;
  bioDraft: string;
  setBioDraft: (value: string) => void;
}

function BrandSubsection({
  nameDraft,
  setNameDraft,
  bioDraft,
  setBioDraft,
}: BrandSubsectionProps) {
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const setBrandName = usePhotographerSettingsStore((s) => s.setBrandName);
  const brandColor = usePhotographerSettingsStore((s) => s.brandColor);
  const setBrandColor = usePhotographerSettingsStore((s) => s.setBrandColor);
  const bio = usePhotographerSettingsStore((s) => s.bio);
  const setBio = usePhotographerSettingsStore((s) => s.setBio);
  const { showToast } = useToast();

  const [saving, setSaving] = useState(false);

  const dirty = nameDraft.trim() !== brandName || bioDraft !== bio;
  const validName =
    nameDraft.trim().length > 0 && nameDraft.trim().length <= BRAND_NAME_MAX;
  const validBio = bioDraft.length <= BIO_MAX;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty || !validName || !validBio) return;
    setSaving(true);
    try {
      // TODO(backend): swap setTimeout for `api.put("/me/photographer/brand", {...})`.
      await new Promise((r) => setTimeout(r, 500));
      setBrandName(nameDraft.trim());
      setBio(bioDraft);
      showToast({ kind: "success", message: "Brand saved." });
    } finally {
      setSaving(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <SubHeading label="Brand" caption="Display name · accent · bio" />
      <div className="space-y-7">
        <div className="flex flex-col gap-2">
          <label htmlFor="brand-name" className="font-sans text-sm text-slate">
            Display name
          </label>
          <input
            id="brand-name"
            value={nameDraft}
            onChange={(e) => setNameDraft(e.target.value)}
            placeholder="Cebu Coastline Photo"
            maxLength={BRAND_NAME_MAX}
            required
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
          />
          <Kicker as="p" tone="soft" tnum>
            {nameDraft.trim().length}/{BRAND_NAME_MAX}
          </Kicker>
        </div>

        <div>
          <p className="font-sans text-sm text-slate mb-3">Accent color</p>
          <div className="flex flex-wrap items-center gap-3">
            {COLOR_ORDER.map((color) => {
              const isActive = brandColor === color;
              return (
                <button
                  key={color}
                  type="button"
                  onClick={() => {
                    setBrandColor(color);
                    if (color !== brandColor) {
                      showToast({
                        kind: "success",
                        message: `Accent set to ${BRAND_COLOR_LABEL[color]}.`,
                      });
                    }
                  }}
                  aria-pressed={isActive}
                  aria-label={BRAND_COLOR_LABEL[color]}
                  className={cn(
                    "relative size-10 rounded-full border-2 transition-all",
                    isActive
                      ? "border-ink scale-110"
                      : "border-line hover:border-slate",
                  )}
                  style={{
                    backgroundColor:
                      color === "none"
                        ? "var(--bone-deep)"
                        : BRAND_COLOR_HEX[color],
                  }}
                >
                  {color === "none" && (
                    <span
                      aria-hidden="true"
                      className="absolute inset-1.5 rounded-full border border-slate-soft border-dashed"
                    />
                  )}
                  {isActive && (
                    <span
                      aria-hidden="true"
                      className="absolute inset-0 rounded-full border-2 border-bone"
                    />
                  )}
                </button>
              );
            })}
          </div>
          <Kicker as="p" tone="soft" tnum className="mt-3">
            {BRAND_COLOR_LABEL[brandColor]}
          </Kicker>
        </div>

        <div className="flex flex-col gap-2">
          <label htmlFor="brand-bio" className="font-sans text-sm text-slate">
            Bio
          </label>
          <textarea
            id="brand-bio"
            value={bioDraft}
            onChange={(e) => setBioDraft(e.target.value)}
            placeholder="Cebu marathon photographer. Three years on the road."
            rows={3}
            maxLength={BIO_MAX}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors resize-none"
          />
          <Kicker
            as="p"
            tone={bioDraft.length > BIO_MAX - 20 ? "default" : "soft"}
            tnum
            className={bioDraft.length > BIO_MAX - 20 ? "text-error" : undefined}
          >
            {bioDraft.length}/{BIO_MAX}
          </Kicker>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="submit"
            disabled={!dirty || !validName || !validBio || saving}
            className="font-sans text-sm font-medium bg-fresh hover:bg-fresh-deep text-bone py-2.5 px-5 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {saving ? "Saving…" : "Save brand"}
            {!saving && <span aria-hidden="true">→</span>}
          </button>
          {dirty && !saving && (
            <button
              type="button"
              onClick={() => {
                setNameDraft(brandName);
                setBioDraft(bio);
              }}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </div>
    </form>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 02 — Watermark
// ─────────────────────────────────────────────────────────────────────────

function WatermarkSlab() {
  const watermark = usePhotographerSettingsStore((s) => s.watermark);
  const setWatermark = usePhotographerSettingsStore((s) => s.setWatermark);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setBusy(true);
    setError(null);
    try {
      // TODO(backend): swap for `api.post("/me/photographer/watermark", formData)`.
      // Server-side will store the PNG and apply it during upload processing
      // so runners see the watermark at gallery render time.
      const dataUrl = await fitToPngDataUrl(file, WATERMARK_MAX_PX);
      setWatermark({ dataUrl, uploadedAt: new Date().toISOString() });
      showToast({ kind: "success", message: "Watermark updated." });
    } catch {
      setError("Could not process this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  function handleRemove() {
    setWatermark(null);
    setError(null);
    showToast({ kind: "success", message: "Watermark removed." });
  }

  return (
    <Slab
      id="watermark"
      number="02"
      title="Watermark"
      caption="Stamped on every photo you upload"
    >
      <div className="space-y-5">
        <p className="font-sans text-sm text-slate max-w-md">
          Runners see your watermark on every photo before they buy — that&apos;s
          how they find you. Use a transparent PNG so it overlays cleanly.
        </p>

        <div className="rounded-2xl overflow-hidden bg-bone-deep border border-line aspect-[3/2] relative">
          <SamplePhoto />
          {watermark && (
            // eslint-disable-next-line @next/next/no-img-element -- data-URL mock; backend will composite watermark onto each photo at upload time.
            <img
              src={watermark.dataUrl}
              alt=""
              className="absolute bottom-4 right-4 max-w-[40%] max-h-[35%] opacity-70 mix-blend-screen pointer-events-none"
              draggable={false}
            />
          )}
          {!watermark && (
            <div className="absolute bottom-4 right-4 px-4 py-2 rounded-full border border-dashed border-line bg-bone/40 backdrop-blur-sm">
              <Kicker as="p">
                Watermark goes here
              </Kicker>
            </div>
          )}
        </div>

        {error && (
          <p role="alert" className="font-sans text-sm text-error">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <label
            className={cn(
              "font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2",
              busy ? "opacity-60 cursor-wait" : "cursor-pointer",
            )}
          >
            {busy
              ? "Processing…"
              : watermark
                ? "Replace watermark"
                : "Upload watermark"}
            {!busy && <span aria-hidden="true">→</span>}
            <input
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handlePick}
              disabled={busy}
              className="sr-only"
            />
          </label>

          {watermark && !busy && (
            <button
              type="button"
              onClick={handleRemove}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
            >
              Remove
            </button>
          )}

          <p className="font-sans text-sm text-slate-soft basis-full md:basis-auto">
            PNG with transparency recommended · downscaled to 600px wide · 8 MB max.
          </p>
        </div>
      </div>
    </Slab>
  );
}

function SamplePhoto() {
  // Bone-deep diagonal gradient as a stand-in for a real race photo. Backend
  // will replace this preview with the photographer's most recent live photo.
  return (
    <div
      className="size-full"
      aria-hidden="true"
      style={{
        background:
          "linear-gradient(135deg, var(--ink-soft) 0%, var(--slate) 45%, var(--bone-deep) 100%)",
      }}
    />
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 03 — Public URL
// ─────────────────────────────────────────────────────────────────────────

function HandleSlab() {
  const handle = usePhotographerSettingsStore((s) => s.handle);
  const setHandle = usePhotographerSettingsStore((s) => s.setHandle);
  const { showToast } = useToast();
  const [draft, setDraft] = useState(handle);
  const [saving, setSaving] = useState(false);

  const validation = validateHandle(draft);
  const dirty = draft.trim().toLowerCase() !== handle;
  const valid = validation === null;

  const previewUrl = draft.trim()
    ? `quickpitik.com/${draft.trim().toLowerCase()}`
    : "quickpitik.com/your-handle";

  const previewHref = valid ? `/${draft.trim().toLowerCase()}` : "#";

  const copy = useCallback(async () => {
    if (!valid) return;
    try {
      await navigator.clipboard.writeText(`https://${previewUrl}`);
      showToast({ kind: "success", message: "Public URL copied." });
    } catch {
      showToast({
        kind: "error",
        message: "Couldn't copy. Try selecting the URL manually.",
      });
    }
  }, [previewUrl, valid, showToast]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty || !valid) return;
    setSaving(true);
    try {
      // TODO(backend): swap setTimeout for `api.put("/me/photographer/handle", { handle })`
      // and check `409 Conflict` for handle already taken across users.
      await new Promise((r) => setTimeout(r, 500));
      setHandle(draft);
      showToast({ kind: "success", message: "Public URL updated." });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Slab
      id="url"
      number="03"
      title="Public URL"
      caption="The address runners visit"
    >
      <form onSubmit={handleSubmit} className="space-y-7">
        <div className="flex flex-col gap-2">
          <label htmlFor="handle" className="font-sans text-sm text-slate">
            Handle
          </label>
          <div className="flex items-baseline gap-2 border-b border-line focus-within:border-fresh transition-colors">
            <span className="font-mono text-base text-slate-soft py-4 pl-1">
              quickpitik.com/
            </span>
            <input
              id="handle"
              value={draft}
              onChange={(e) =>
                setDraft(
                  e.target.value
                    .replace(/[^a-zA-Z0-9-]/g, "")
                    .toLowerCase()
                    .slice(0, 32),
                )
              }
              placeholder="your-handle"
              autoComplete="off"
              spellCheck={false}
              required
              className="flex-1 min-w-0 bg-transparent focus:outline-none py-4 text-lg font-mono text-ink placeholder:text-slate-soft"
            />
          </div>
          <p
            className={cn(
              "font-sans text-sm",
              draft.length === 0
                ? "text-slate-soft"
                : valid
                  ? "text-slate"
                  : "text-error",
            )}
          >
            {draft.length === 0
              ? "Lowercase letters, numbers, and dashes. 3–32 characters."
              : (validation ?? "Looks good — you can save.")}
          </p>
        </div>

        <div className="border border-line rounded-2xl bg-bone-deep/40 px-5 py-4 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="min-w-0">
            <Kicker as="p" tone="soft">
              Live URL
            </Kicker>
            <p className="font-mono text-base md:text-lg text-ink mt-2 break-all">
              {previewUrl}
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 shrink-0">
            <button
              type="button"
              onClick={copy}
              disabled={!valid}
              className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors disabled:opacity-40 disabled:cursor-not-allowed disabled:no-underline"
            >
              Copy link
            </button>
            <a
              href={previewHref}
              target="_blank"
              rel="noopener noreferrer"
              aria-disabled={!valid}
              onClick={(e) => {
                if (!valid) e.preventDefault();
              }}
              className={cn(
                "font-sans text-sm transition-colors inline-flex items-center gap-1",
                valid
                  ? "text-ink hover:text-fresh"
                  : "text-slate-soft cursor-not-allowed",
              )}
            >
              Preview
              <span aria-hidden="true">↗</span>
            </a>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <button
            type="submit"
            disabled={!dirty || !valid || saving}
            className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
          >
            {saving ? "Saving…" : "Save URL"}
            {!saving && <span aria-hidden="true">→</span>}
          </button>
          {dirty && !saving && (
            <button
              type="button"
              onClick={() => setDraft(handle)}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
            >
              Reset
            </button>
          )}
        </div>
      </form>
    </Slab>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 04 — Region
// ─────────────────────────────────────────────────────────────────────────

function RegionSlab() {
  const region = usePhotographerSettingsStore((s) => s.region);
  const setRegion = usePhotographerSettingsStore((s) => s.setRegion);
  const { showToast } = useToast();

  const selectedRegion = region ? getRegion(region.regionCode) : undefined;
  const selectedProvince =
    region && selectedRegion
      ? selectedRegion.provinces.find((p) => p.code === region.provinceCode)
      : undefined;

  function handlePickRegion(regionCode: string) {
    if (region?.regionCode === regionCode) return;
    // Reset province when region changes — the user must re-pick.
    const next = getRegion(regionCode);
    if (!next) return;
    setRegion({ regionCode, provinceCode: "" });
  }

  function handlePickProvince(provinceCode: string) {
    if (!region) return;
    if (region.provinceCode === provinceCode) return;
    setRegion({ regionCode: region.regionCode, provinceCode });
    const summary = formatRegionLabel(region.regionCode, provinceCode);
    if (summary) {
      showToast({ kind: "success", message: `Region set · ${summary}` });
    }
  }

  return (
    <Slab
      id="region"
      number="04"
      title="Region"
      caption="Where you cover events"
    >
      <div className="space-y-7">
        <p className="font-sans text-sm text-slate max-w-md">
          Helps runners surface local photographers and lets organizers invite
          the right talent for their events.
        </p>

        <div className="grid md:grid-cols-2 gap-5 md:gap-7">
          <div className="flex flex-col gap-2">
            <span className="font-sans text-sm text-slate">Region</span>
            <Dropdown
              align="left"
              ariaLabel="Pick a region"
              className="w-full"
              panelClassName="max-h-80 overflow-y-auto w-[min(28rem,calc(100vw-2rem))]"
              trigger={
                <span className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left transition-colors hover:border-slate">
                  <span
                    className={cn(
                      "truncate text-lg",
                      selectedRegion ? "text-ink" : "text-slate-soft",
                    )}
                  >
                    {selectedRegion?.name ?? "Pick a region"}
                  </span>
                  <span aria-hidden="true" className="text-slate shrink-0">
                    ▾
                  </span>
                </span>
              }
            >
              {REGION_GROUP_ORDER.map((group) => (
                <div key={group} className="pt-2 first:pt-0">
                  <Kicker as="p" tone="soft" className="px-4 py-2">
                    {REGION_GROUP_LABEL[group]}
                  </Kicker>
                  {PH_REGIONS.filter((r) => r.group === group).map((r) => (
                    <DropdownItem
                      key={r.code}
                      onClick={() => handlePickRegion(r.code)}
                      active={region?.regionCode === r.code}
                    >
                      {r.name}
                    </DropdownItem>
                  ))}
                </div>
              ))}
            </Dropdown>
          </div>

          <div className="flex flex-col gap-2">
            <span className="font-sans text-sm text-slate">Province</span>
            {selectedRegion ? (
              <Dropdown
                align="left"
                ariaLabel="Pick a province"
                className="w-full"
                panelClassName="max-h-80 overflow-y-auto w-[min(20rem,calc(100vw-2rem))]"
                trigger={
                  <span className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left transition-colors hover:border-slate">
                    <span
                      className={cn(
                        "truncate text-lg",
                        selectedProvince ? "text-ink" : "text-slate-soft",
                      )}
                    >
                      {selectedProvince?.name ?? "Pick a province"}
                    </span>
                    <span aria-hidden="true" className="text-slate shrink-0">
                      ▾
                    </span>
                  </span>
                }
              >
                {selectedRegion.provinces.map((p) => (
                  <DropdownItem
                    key={p.code}
                    onClick={() => handlePickProvince(p.code)}
                    active={region?.provinceCode === p.code}
                  >
                    {p.name}
                  </DropdownItem>
                ))}
              </Dropdown>
            ) : (
              <span className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left text-lg text-slate-soft cursor-not-allowed">
                <span className="truncate">Pick a region first</span>
                <span aria-hidden="true">▾</span>
              </span>
            )}
          </div>
        </div>

        {region && selectedProvince && (
          <Kicker as="p" tnum>
            Selected · {selectedProvince.name} · {selectedRegion?.shortName}
          </Kicker>
        )}
      </div>
    </Slab>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 05 — Social & verification
// ─────────────────────────────────────────────────────────────────────────

function SocialSlab() {
  const socials = usePhotographerSettingsStore((s) => s.socials);
  const addSocial = usePhotographerSettingsStore((s) => s.addSocial);
  const updateSocial = usePhotographerSettingsStore((s) => s.updateSocial);
  const removeSocial = usePhotographerSettingsStore((s) => s.removeSocial);
  const { showToast } = useToast();

  function handleAdd(platform: SocialPlatform) {
    addSocial(platform, "");
    showToast({
      kind: "info",
      message: `${SOCIAL_PLATFORM_LABEL[platform]} row added — drop your link in.`,
    });
  }

  function handleRemove(id: string, platform: SocialPlatform) {
    removeSocial(id);
    showToast({
      kind: "success",
      message: `${SOCIAL_PLATFORM_LABEL[platform]} link removed.`,
    });
  }

  const filledCount = socials.filter((s) => s.url.trim().length > 0).length;

  return (
    <Slab
      id="social"
      number="05"
      title="Social & verification"
      caption="A live link is the cheapest way runners check you're real"
    >
      <div className="space-y-7">
        <p className="font-sans text-sm text-slate max-w-md">
          At least one social link is required. Most Cebu marathon photographers
          use Facebook — add Instagram or TikTok too if that&apos;s where your
          gallery lives.
        </p>

        {socials.length > 0 && (
          <ul className="space-y-4">
            {socials.map((link) => (
              <SocialRow
                key={link.id}
                link={link}
                onChange={(url) => updateSocial(link.id, url)}
                onRemove={() => handleRemove(link.id, link.platform)}
              />
            ))}
          </ul>
        )}

        <div>
          <Kicker as="p" tone="soft">
            Add a platform
          </Kicker>
          <div className="mt-3 flex flex-wrap gap-2.5">
            {SOCIAL_PLATFORMS.map((platform) => (
              <button
                key={platform}
                type="button"
                onClick={() => handleAdd(platform)}
                className="font-sans text-sm py-2 px-4 rounded-full border border-line text-ink hover:bg-bone-deep/60 hover:border-slate transition-colors inline-flex items-center gap-2"
              >
                <Kicker>
                  {SOCIAL_PLATFORM_TILE[platform]}
                </Kicker>
                {SOCIAL_PLATFORM_LABEL[platform]}
              </button>
            ))}
          </div>
        </div>

        {socials.length > 0 && (
          <Kicker as="p" tone="soft" tnum>
            {filledCount} of {socials.length} filled in
          </Kicker>
        )}
      </div>
    </Slab>
  );
}

interface SocialRowProps {
  link: SocialLink;
  onChange: (url: string) => void;
  onRemove: () => void;
}

function SocialRow({ link, onChange, onRemove }: SocialRowProps) {
  const placeholder: Record<SocialPlatform, string> = {
    facebook: "https://facebook.com/your-page",
    instagram: "https://instagram.com/your-handle",
    tiktok: "https://tiktok.com/@your-handle",
    x: "https://x.com/your-handle",
    youtube: "https://youtube.com/@your-channel",
    website: "https://your-site.com",
  };

  return (
    <li className="flex items-start gap-4">
      <Kicker
        aria-hidden="true"
        className="size-10 shrink-0 rounded-2xl border border-line bg-bone-deep/60 flex items-center justify-center text-ink"
      >
        {SOCIAL_PLATFORM_TILE[link.platform]}
      </Kicker>
      <div className="flex-1 min-w-0 flex flex-col gap-2">
        <div className="flex items-baseline justify-between gap-3">
          <span className="font-sans text-sm text-slate">
            {SOCIAL_PLATFORM_LABEL[link.platform]}
          </span>
          <button
            type="button"
            onClick={onRemove}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
          >
            Remove
          </button>
        </div>
        <input
          type="url"
          value={link.url}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder[link.platform]}
          className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
        />
      </div>
    </li>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 06 — Payout accounts
// ─────────────────────────────────────────────────────────────────────────

function validatePayoutNumber(method: PayoutMethod, raw: string): string | null {
  const d = raw.replace(/\D/g, "");
  if (method === "gcash" || method === "maya") {
    if (d.length === 0) return null;
    if (d.length !== 11) return "Must be 11 digits.";
    if (!d.startsWith("09")) return "Must start with 09.";
    return null;
  }
  if (d.length === 0) return null;
  if (d.length < 10) return "Must be at least 10 digits.";
  return null;
}

function PayoutSlab() {
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  const addPayout = usePhotographerSettingsStore((s) => s.addPayout);
  const setPrimaryPayout = usePhotographerSettingsStore(
    (s) => s.setPrimaryPayout,
  );
  const removePayout = usePhotographerSettingsStore((s) => s.removePayout);
  const { showToast } = useToast();

  const [draftMethod, setDraftMethod] = useState<PayoutMethod | null>(null);

  function handlePickMethod(method: PayoutMethod) {
    setDraftMethod((current) => (current === method ? null : method));
  }

  function handleAdd(input: {
    method: PayoutMethod;
    accountNumber: string;
    accountName: string;
    qr: { dataUrl: string; uploadedAt: string } | null;
  }) {
    addPayout(input);
    setDraftMethod(null);
    showToast({
      kind: "success",
      message: `${PAYOUT_METHOD_LABEL[input.method]} account added${
        payouts.length === 0 ? " · set as primary" : ""
      }.`,
    });
  }

  function handleMakePrimary(id: string, method: PayoutMethod) {
    setPrimaryPayout(id);
    showToast({
      kind: "success",
      message: `${PAYOUT_METHOD_LABEL[method]} is now your primary payout.`,
    });
  }

  function handleRemove(id: string, method: PayoutMethod) {
    removePayout(id);
    showToast({
      kind: "success",
      message: `${PAYOUT_METHOD_LABEL[method]} account removed.`,
    });
  }

  return (
    <Slab
      id="payout"
      number="06"
      title="Payout accounts"
      caption="Where we send your sales"
    >
      <div className="space-y-7">
        <p className="font-sans text-sm text-slate max-w-md">
          Only seen by QuickPitik finance. One account is your primary — the
          rest are backups in case the primary is unavailable. Make sure the
          account name matches what&apos;s registered or the transfer will
          bounce.
        </p>

        <div>
          <Kicker as="p" tone="soft">
            Add an account
          </Kicker>
          <div className="mt-3 flex flex-wrap gap-2.5">
            {PAYOUT_METHODS.map((method) => {
              const active = draftMethod === method;
              return (
                <button
                  key={method}
                  type="button"
                  onClick={() => handlePickMethod(method)}
                  aria-pressed={active}
                  className={cn(
                    "font-sans text-sm py-2 px-4 rounded-full border transition-colors inline-flex items-center gap-2",
                    active
                      ? "bg-ink text-bone border-ink"
                      : "border-line text-ink hover:bg-bone-deep/60 hover:border-slate",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className="size-2 rounded-full"
                    style={{ backgroundColor: PAYOUT_METHOD_HEX[method] }}
                  />
                  {PAYOUT_METHOD_LABEL[method]}
                </button>
              );
            })}
          </div>
        </div>

        {draftMethod && (
          <PayoutForm
            method={draftMethod}
            onCancel={() => setDraftMethod(null)}
            onSave={handleAdd}
          />
        )}

        {payouts.length > 0 && (
          <div className="space-y-3">
            <Kicker as="p" tone="soft">
              Your accounts
            </Kicker>
            <ul className="space-y-3">
              {payouts.map((account) => (
                <PayoutCard
                  key={account.id}
                  account={account}
                  onMakePrimary={() =>
                    handleMakePrimary(account.id, account.method)
                  }
                  onRemove={() => handleRemove(account.id, account.method)}
                />
              ))}
            </ul>
            <Kicker as="p" tone="soft" tnum>
              {payouts.length} account{payouts.length === 1 ? "" : "s"} ·{" "}
              {payouts.find((p) => p.isPrimary)
                ? `${PAYOUT_METHOD_LABEL[payouts.find((p) => p.isPrimary)!.method]} primary`
                : "no primary set"}
            </Kicker>
          </div>
        )}
      </div>
    </Slab>
  );
}

interface PayoutFormProps {
  method: PayoutMethod;
  onCancel: () => void;
  onSave: (input: {
    method: PayoutMethod;
    accountNumber: string;
    accountName: string;
    qr: { dataUrl: string; uploadedAt: string } | null;
  }) => void;
}

function PayoutForm({ method, onCancel, onSave }: PayoutFormProps) {
  const { showToast } = useToast();
  const [accountName, setAccountName] = useState("");
  const [accountNumber, setAccountNumber] = useState("");
  const [qr, setQr] = useState<{ dataUrl: string; uploadedAt: string } | null>(
    null,
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const numberError = validatePayoutNumber(method, accountNumber);
  const validNumber =
    accountNumber.replace(/\D/g, "").length > 0 && numberError === null;
  const validName = accountName.trim().length > 0;
  const canSave = validNumber && validName;

  const numberLabel = method === "gotyme" ? "Account number" : "Mobile number";
  const numberPlaceholder =
    method === "gotyme" ? "1234 5678 9012" : "0917 555 0101";

  async function handleQrPick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setBusy(true);
    setError(null);
    try {
      // TODO(backend): swap for `api.post("/me/photographer/payouts/{id}/qr", formData)`
      // — server stores the image privately; only finance staff can read it.
      const dataUrl = await fitToPngDataUrl(file, QR_MAX_PX);
      setQr({ dataUrl, uploadedAt: new Date().toISOString() });
    } catch {
      setError("Could not process this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSave) return;
    onSave({
      method,
      accountName: accountName.trim(),
      accountNumber: accountNumber.replace(/\D/g, ""),
      qr,
    });
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="border border-line rounded-2xl bg-bone-deep/30 p-5 md:p-6 space-y-6"
    >
      <div className="flex items-center justify-between gap-3">
        <p className="font-display text-lg md:text-xl font-medium tracking-tight text-ink inline-flex items-center gap-2">
          <span
            aria-hidden="true"
            className="size-2 rounded-full"
            style={{ backgroundColor: PAYOUT_METHOD_HEX[method] }}
          />
          New {PAYOUT_METHOD_LABEL[method]} account
        </p>
        <button
          type="button"
          onClick={() => {
            onCancel();
            showToast({ kind: "info", message: "New account discarded." });
          }}
          className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
        >
          Cancel
        </button>
      </div>

      <div className="grid md:grid-cols-2 gap-5 md:gap-6">
        <div className="flex flex-col gap-2">
          <label
            htmlFor={`payout-name-${method}`}
            className="font-sans text-sm text-slate"
          >
            Account name
          </label>
          <input
            id={`payout-name-${method}`}
            value={accountName}
            onChange={(e) => setAccountName(e.target.value)}
            placeholder="Juan Dela Cruz"
            maxLength={ACCOUNT_NAME_MAX}
            required
            autoComplete="off"
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <p className="font-sans text-sm text-slate-soft">
            Must match the name on your {PAYOUT_METHOD_LABEL[method]} account.
          </p>
        </div>

        <div className="flex flex-col gap-2">
          <label
            htmlFor={`payout-number-${method}`}
            className="font-sans text-sm text-slate"
          >
            {numberLabel}
          </label>
          <input
            id={`payout-number-${method}`}
            value={formatPayoutNumber(method, accountNumber)}
            onChange={(e) => setAccountNumber(e.target.value)}
            placeholder={numberPlaceholder}
            inputMode="numeric"
            required
            autoComplete="off"
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base font-mono tnum text-ink placeholder:text-slate-soft transition-colors"
          />
          <p
            className={cn(
              "font-sans text-sm",
              numberError ? "text-error" : "text-slate-soft",
            )}
          >
            {numberError ?? (method === "gotyme"
              ? "Your GoTyme account number, digits only."
              : "11-digit mobile number registered to this wallet.")}
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <span className="font-sans text-sm text-slate">QR code (optional)</span>
        <div className="flex flex-col sm:flex-row sm:items-start gap-4 sm:gap-5">
          <div className="size-32 md:size-36 shrink-0 rounded-2xl border border-line bg-bone overflow-hidden flex items-center justify-center">
            {qr ? (
              // eslint-disable-next-line @next/next/no-img-element -- data-URL mock; backend stores the QR privately.
              <img
                src={qr.dataUrl}
                alt=""
                className="size-full object-contain"
                draggable={false}
              />
            ) : (
              <Kicker as="p" tone="soft" className="text-center px-2">
                QR preview
              </Kicker>
            )}
          </div>
          <div className="flex flex-col gap-3">
            <label
              className={cn(
                "self-start font-sans text-sm border border-ink text-ink hover:bg-ink hover:text-bone py-2 px-5 rounded-full transition-colors inline-flex items-center gap-2",
                busy ? "opacity-60 cursor-wait" : "cursor-pointer",
              )}
            >
              {busy ? "Processing…" : qr ? "Replace QR" : "Upload QR"}
              {!busy && <span aria-hidden="true">→</span>}
              <input
                type="file"
                accept={ACCEPTED_IMAGE_MIME.join(",")}
                onChange={handleQrPick}
                disabled={busy}
                className="sr-only"
              />
            </label>
            {qr && !busy && (
              <button
                type="button"
                onClick={() => setQr(null)}
                className="self-start font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
              >
                Remove QR
              </button>
            )}
            <p className="font-sans text-sm text-slate-soft max-w-xs">
              From the {PAYOUT_METHOD_LABEL[method]} app: open your QR, save the
              image to your gallery, then upload it here. Backed up so finance
              can scan if a transfer needs to retry.
            </p>
          </div>
        </div>

        {error && (
          <p role="alert" className="font-sans text-sm text-error">
            {error}
          </p>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 pt-2">
        <button
          type="submit"
          disabled={!canSave}
          className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
        >
          Save account
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </form>
  );
}

interface PayoutCardProps {
  account: PayoutAccount;
  onMakePrimary: () => void;
  onRemove: () => void;
}

function PayoutCard({ account, onMakePrimary, onRemove }: PayoutCardProps) {
  const formatted = formatPayoutNumber(account.method, account.accountNumber);

  return (
    <li className="border border-line rounded-2xl bg-bone p-5 flex flex-col md:flex-row gap-5 md:items-start md:justify-between">
      <div className="flex items-start gap-4 min-w-0">
        <span
          aria-hidden="true"
          className="size-10 shrink-0 rounded-2xl border border-line flex items-center justify-center"
          style={{ backgroundColor: `${PAYOUT_METHOD_HEX[account.method]}1A` }}
        >
          <span
            className="size-2.5 rounded-full"
            style={{ backgroundColor: PAYOUT_METHOD_HEX[account.method] }}
          />
        </span>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <p className="font-display text-lg md:text-xl font-medium tracking-tight text-ink">
              {PAYOUT_METHOD_LABEL[account.method]}
            </p>
            {account.isPrimary && (
              <Kicker tone="active" className="inline-flex items-center gap-1.5">
                <span
                  aria-hidden="true"
                  className="size-1.5 rounded-full bg-fresh"
                />
                Primary
              </Kicker>
            )}
          </div>
          <p className="font-sans text-sm text-slate mt-1 truncate">
            {account.accountName || "—"}
          </p>
          <p className="font-mono tnum text-base text-ink mt-2 break-all">
            {formatted || "—"}
          </p>
          {account.qr && (
            <Kicker as="p" tone="soft" className="mt-3">
              QR uploaded
            </Kicker>
          )}
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 shrink-0 md:flex-col md:items-end">
        {!account.isPrimary && (
          <button
            type="button"
            onClick={onMakePrimary}
            className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            Make primary
          </button>
        )}
        <button
          type="button"
          onClick={onRemove}
          className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
        >
          Remove
        </button>
      </div>
    </li>
  );
}
