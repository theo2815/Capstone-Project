"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type FormEvent,
} from "react";
import { createPortal } from "react-dom";
import { Slab } from "@/components/profile-shell";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { CharCounter } from "@/components/ui/char-counter";
import { Dropdown, DropdownItem } from "@/components/ui/dropdown";
import { FieldError } from "@/components/ui/field-error";
import { Kicker } from "@/components/ui/kicker";
import { Modal } from "@/components/ui/modal";
import { useAuth } from "@/hooks/use-auth";
import { useConfirmation } from "@/hooks/use-confirmation";
import { useDebouncedSave } from "@/hooks/use-debounced-save";
import { useToast } from "@/hooks/use-toast";
import { uploadAvatar } from "@/lib/api-avatar";
import {
  postCover,
  postWatermark,
  putBrand,
  putHandle,
  putRegion,
  submitVerification,
  withdrawVerification,
} from "@/lib/api-photographer-settings";
import { usePhotographerVerificationSync } from "@/lib/photographer-verification-sync";
import { useAdminUserStore } from "@/store/admin-user-store";
import { useAuthStore } from "@/store/auth-store";
import { useUserMediaStore } from "@/store/user-media-store";
import type { AdminUserRow } from "@/lib/admin-user-registry";
import {
  ACCEPTED_IMAGE_MIME,
  ImageProcessingError,
  fitToDataUrl,
  fitToPngDataUrl,
  inspectImageDimensions,
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
  type VerificationStatus,
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

// Cover banner is 16:5 (= 3.2). Anything below this aspect ratio is portrait
// or near-square — the middle horizontal strip is the only thing that ends up
// visible after the cover-fit, so warn the user before committing. Threshold
// is intentionally wider than 3.2 so we only flag obviously-wrong shapes, not
// just imperfect crops.
const COVER_MIN_ASPECT = 1.4;

// Fire-and-forget for photographer-settings API calls. Local store updates
// apply immediately for UI snap; API call runs in the background. On
// failure we log to console — backend is source of truth on next refresh.
function fireSettings(label: string, p: Promise<unknown>): void {
  void p.catch((err) => {
    console.error(`[photographer/settings] ${label} backend call failed`, err);
  });
}

// Mirror a fresh "Submit for review" into the admin store so the
// /admin/verifications queue shows the row. Phase F backend will receive
// the submit POST and emit a server-rendered row on the admin's next poll;
// in mock mode, we have to bridge the two stores ourselves. Returns null
// when there's no authenticated session (defensive — caller bails).
function buildAdminSubmissionRow(): AdminUserRow | null {
  const sessionUser = useAuthStore.getState().user;
  if (!sessionUser) return null;
  const settings = usePhotographerSettingsStore.getState();
  const regionLabel = settings.region
    ? formatRegionLabel(settings.region.regionCode, settings.region.provinceCode)
    : null;
  return {
    userId: sessionUser.id,
    role: "PHOTOGRAPHER",
    email: sessionUser.email,
    name: sessionUser.name,
    brandName: settings.brandName.trim() || null,
    handle: settings.handle.trim() || null,
    region: regionLabel,
    // City isn't a separate field in photographer-settings yet — fall back to
    // the region label so the admin row still shows a location string.
    city: regionLabel ?? "—",
    createdAt: sessionUser.createdAt,
    verificationStatus: "pending",
    suspendedAt: null,
    suspensionReason: null,
    settingsSnapshot: {
      hasCover: !!settings.cover,
      hasBrandName: settings.brandName.trim().length > 0,
      hasWatermark: !!settings.watermark,
      hasHandle: settings.handle.trim().length >= 3,
      hasRegion: !!settings.region,
      socialCount: settings.socials.filter((s) => s.url.trim().length > 0)
        .length,
      payoutCount: settings.payouts.length,
    },
  };
}

function mirrorSubmissionToAdminStore(): void {
  const row = buildAdminSubmissionRow();
  if (!row) return;
  useAdminUserStore.getState().upsertSubmission(row);
}

// Phase 1 — consume the BE submitVerification response instead of fire-and-
// forget. Three outcomes per the controller + DTO:
//   1. { status: "pending" }                 — accepted; row in DB is PENDING.
//   2. { status: "incomplete", missing[] }   — soft-reject; nothing written.
//   3. throws (ApiError or network)          — hard failure; nothing written.
// Returns whether the BE accepted, so callers (the nudge modal) can decide
// to close their own UI or stay open for a retry.
async function submitForReview(opts: {
  setStatus: (s: VerificationStatus) => void;
  showToast: (t: { kind: "info" | "success" | "error"; message: string }) => void;
}): Promise<{ submitted: boolean }> {
  try {
    const response = await submitVerification();
    if (response.status === "pending") {
      opts.setStatus("pending");
      mirrorSubmissionToAdminStore();
      opts.showToast({
        kind: "info",
        message: "Submitted for review. We'll let you know.",
      });
      return { submitted: true };
    }
    if (response.status === "incomplete") {
      const missingLabel =
        response.missing && response.missing.length > 0
          ? response.missing.join(", ")
          : "some required fields";
      opts.showToast({
        kind: "error",
        message: `Can't submit yet — missing ${missingLabel}.`,
      });
      return { submitted: false };
    }
    // Defensive: BE shouldn't return "approved" from /verification, but if it
    // ever does, sync local state so the dashboard banner reflects it.
    opts.setStatus(response.status);
    return { submitted: false };
  } catch (err) {
    console.error("[photographer/settings] submitVerification failed", err);
    opts.showToast({
      kind: "error",
      message: "Couldn't submit — check your connection and try again.",
    });
    return { submitted: false };
  }
}

// Map our typed image-processing errors to user-facing copy. Falls back to a
// generic message when the error is a plain Error or unknown.
function imageErrorCopy(err: unknown): string {
  if (err instanceof ImageProcessingError) {
    if (err.code === "decode") {
      return "This file looks corrupted. Try exporting a fresh JPEG or PNG.";
    }
    if (err.code === "canvas-tainted") {
      return "Browser blocked this image. Try a different file.";
    }
  }
  return "Could not process this image. Try another.";
}

// Per-platform URL shape gates for Slab 05. Loose enough to cover the common
// subdomains (www, m, web, fb.com short form, youtu.be) but strict enough to
// catch a TikTok URL pasted into the Facebook row. The website row only
// requires a valid http(s) prefix + a host with a dot.
const SOCIAL_URL_PATTERN: Record<SocialPlatform, RegExp> = {
  facebook: /^https?:\/\/(?:www\.|m\.|web\.)?(?:facebook\.com|fb\.com|fb\.me)\//i,
  instagram: /^https?:\/\/(?:www\.)?instagram\.com\//i,
  tiktok: /^https?:\/\/(?:www\.|vm\.)?tiktok\.com\//i,
  x: /^https?:\/\/(?:www\.)?(?:x\.com|twitter\.com)\//i,
  youtube: /^https?:\/\/(?:www\.|m\.)?(?:youtube\.com|youtu\.be)\//i,
  website: /^https?:\/\/[^\s/]+\.[^\s/]+/i,
};

// 2 KB is well past any real social URL but stops a megabyte-paste from
// reaching the store and the eventual backend.
const SOCIAL_URL_MAX = 2048;

// Strip protocol, trailing slash, and case for duplicate detection — so
// `https://Facebook.com/foo/` and `http://facebook.com/foo` collide.
function normalizeSocialUrl(url: string): string {
  return url.trim().toLowerCase().replace(/^https?:\/\//, "").replace(/\/+$/, "");
}

interface SocialValidationContext {
  duplicateUrls: ReadonlySet<string>;
}

function validateSocialUrl(
  platform: SocialPlatform,
  url: string,
  ctx: SocialValidationContext,
): string | null {
  const trimmed = url.trim();
  if (trimmed.length === 0) return "Add a URL or remove this row.";
  if (trimmed.length > SOCIAL_URL_MAX) {
    return `URL is at most ${SOCIAL_URL_MAX} characters.`;
  }
  if (!SOCIAL_URL_PATTERN[platform].test(trimmed)) {
    if (platform === "website") {
      return "URL must start with http:// or https://.";
    }
    return `This doesn't look like a ${SOCIAL_PLATFORM_LABEL[platform]} URL.`;
  }
  if (ctx.duplicateUrls.has(normalizeSocialUrl(trimmed))) {
    return "This URL is already added.";
  }
  return null;
}

export default function SettingsPage() {
  // Phase 4 — keep verificationStatus in sync with the BE so admin
  // approve/reject decisions show up here without a manual refresh.
  // Pulls on mount, on focus, and every 30s while pending.
  usePhotographerVerificationSync();

  return (
    <>
      <PendingEditWatcher />
      <VerificationStatusPanel />
      <PublicProfileSlab />
      <WatermarkSlab />
      <HandleSlab />
      <RegionSlab />
      <SocialSlab />
      <PayoutSlab />
      <ReadyToSubmitNudge />
    </>
  );
}

// While status === "pending", keep the admin store's snapshot in sync with
// the photographer's live edits, and auto-flip back to "incomplete" if any
// required field is removed. The photographer is allowed to keep editing
// after they've submitted (a sealed-envelope model would be cleaner but
// makes "I have a typo, can I fix it?" require admin support churn), so
// the admin always sees the latest snapshot. If they delete a required
// field while pending, the application can't stay pending — drop it from
// the admin queue and put the photographer back in the "Not yet verified"
// state so they re-submit when fixed.
function PendingEditWatcher() {
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const setStatus = usePhotographerSettingsStore(
    (s) => s.setVerificationStatus,
  );
  const complete = useAllRequiredFilled();
  // Subscribe to every snapshot-contributing field so this effect re-fires
  // on any settings change. The `useAllRequiredFilled` boolean alone isn't
  // enough — if the photographer edits a non-gating detail (brand name,
  // bio, social URL) the boolean stays true and the effect wouldn't
  // re-mirror without these explicit subscriptions.
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const bio = usePhotographerSettingsStore((s) => s.bio);
  const brandColor = usePhotographerSettingsStore((s) => s.brandColor);
  const watermark = usePhotographerSettingsStore((s) => s.watermark);
  const handle = usePhotographerSettingsStore((s) => s.handle);
  const region = usePhotographerSettingsStore((s) => s.region);
  const socials = usePhotographerSettingsStore((s) => s.socials);
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  const avatar = useUserMediaStore((s) => s.avatar);

  useEffect(() => {
    if (status !== "pending") return;
    const sessionUser = useAuthStore.getState().user;
    if (!sessionUser) return;
    if (!complete) {
      useAdminUserStore.getState().revokeSubmission(sessionUser.id);
      setStatus("incomplete");
      return;
    }
    mirrorSubmissionToAdminStore();
  }, [
    status,
    complete,
    cover,
    brandName,
    bio,
    brandColor,
    watermark,
    handle,
    region,
    socials,
    payouts,
    avatar,
    setStatus,
  ]);

  return null;
}

// Auto-opens once when the photographer fills the last required field, so
// they don't have to scroll back to the top to find the Submit button.
// Dismissing the modal replaces it with a sticky pill bottom-right that
// stays visible until they submit (or un-complete the form, in which case
// the nudge resets so it can fire again on next completion).
function ReadyToSubmitNudge() {
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const setStatus = usePhotographerSettingsStore(
    (s) => s.setVerificationStatus,
  );
  const { showToast } = useToast();

  const complete = useAllRequiredFilled();
  const [modalOpen, setModalOpen] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const wasCompleteRef = useRef(false);

  useEffect(() => {
    if (status !== "incomplete") return;
    const wasComplete = wasCompleteRef.current;
    wasCompleteRef.current = complete;
    // Auto-open on the rising edge only (false → true). If the photographer
    // dismissed already and the form is still complete, don't re-open.
    if (!wasComplete && complete && !dismissed) {
      setModalOpen(true);
    }
    // Re-arm: if they drop below complete after dismissing, allow the modal
    // to fire again the next time they fill everything in.
    if (!complete && dismissed) {
      setDismissed(false);
    }
  }, [complete, status, dismissed]);

  // Clear nudge state when the photographer leaves the incomplete branch
  // (submitted for review, or admin approved).
  useEffect(() => {
    if (status !== "incomplete") {
      setModalOpen(false);
      setDismissed(false);
      wasCompleteRef.current = false;
    }
  }, [status]);

  async function handleSubmit() {
    if (submitting) return;
    setSubmitting(true);
    try {
      const result = await submitForReview({ setStatus, showToast });
      // Only dismiss the modal on accepted submit. On incomplete/network
      // failure, keep it open so the photographer can retry or see context
      // before navigating away.
      if (result.submitted) setModalOpen(false);
    } finally {
      setSubmitting(false);
    }
  }

  function handleDismiss() {
    setDismissed(true);
    setModalOpen(false);
  }

  if (status !== "incomplete") return null;

  const showPill = complete && dismissed && !modalOpen;

  return (
    <>
      <Modal isOpen={modalOpen} onClose={handleDismiss} title="Ready for review">
        <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
          You&apos;ve filled everything in.
        </p>
        <p className="font-sans text-base text-slate mt-3 max-w-md">
          Submit your settings so we can verify you. Reviews take 1–2 business
          days. You&apos;ll be able to upload as soon as we&apos;re done.
        </p>
        <div className="mt-7 flex flex-wrap items-center gap-4">
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            {submitting ? "Submitting…" : "Submit for review"}
            <span aria-hidden="true">→</span>
          </button>
          <button
            type="button"
            onClick={handleDismiss}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
          >
            Maybe later
          </button>
        </div>
      </Modal>
      {showPill && <ReadyToSubmitPill onClick={() => setModalOpen(true)} />}
    </>
  );
}

function ReadyToSubmitPill({ onClick }: { onClick: () => void }) {
  // Portals to body so it escapes any transform-context ancestor (mirrors the
  // Modal/Dropdown portal pattern — see notes/ui-pitfalls.md).
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);
  if (!mounted) return null;
  return createPortal(
    <button
      type="button"
      onClick={onClick}
      className="fixed bottom-6 right-6 z-50 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-5 rounded-full shadow-2xl transition-colors inline-flex items-center gap-2"
    >
      <span aria-hidden="true" className="size-2 rounded-full bg-bone" />
      Submit for review
      <span aria-hidden="true">→</span>
    </button>,
    document.body,
  );
}

// Subscribes to every required field so consumers re-render whenever any
// changes. The store's `isComplete()` getter alone won't trigger updates —
// Zustand only fires re-renders when the subscribed slice (the function
// reference) changes, not when the underlying fields change. Profile
// picture lives in useUserMediaStore (shared with the runner account flow).
function useAllRequiredFilled(): boolean {
  const avatar = useUserMediaStore((s) => s.avatar);
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const watermark = usePhotographerSettingsStore((s) => s.watermark);
  const handle = usePhotographerSettingsStore((s) => s.handle);
  const region = usePhotographerSettingsStore((s) => s.region);
  const socials = usePhotographerSettingsStore((s) => s.socials);
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  return (
    avatar !== null &&
    cover !== null &&
    brandName.trim().length > 0 &&
    watermark !== null &&
    handle.trim().length > 0 &&
    region !== null &&
    socials.some((sl) => sl.url.trim().length > 0) &&
    payouts.some((p) => p.isPrimary)
  );
}

function VerificationStatusPanel() {
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const setStatus = usePhotographerSettingsStore(
    (s) => s.setVerificationStatus,
  );
  const { showToast } = useToast();
  const { confirm } = useConfirmation();
  const complete = useAllRequiredFilled();
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit() {
    if (submitting) return;
    setSubmitting(true);
    try {
      // Admin approval is the only path to "approved" — handled in
      // `/admin/verifications`. The photographer sits in `pending` until a
      // human reviews them. Phase 1 wires the real BE call: the helper
      // flips local state only after the BE confirms.
      await submitForReview({ setStatus, showToast });
    } finally {
      setSubmitting(false);
    }
  }

  async function handleWithdraw() {
    const ok = await confirm({
      title: "Withdraw application?",
      message:
        "Your submission will be removed from the admin queue and your status returns to incomplete. Your settings stay saved — you can edit and resubmit anytime.",
      confirmLabel: "Withdraw",
      cancelLabel: "Keep waiting",
      danger: true,
    });
    if (!ok) return;
    try {
      const response = await withdrawVerification();
      const sessionUser = useAuthStore.getState().user;
      if (sessionUser) {
        // Drop the optimistic submission row from the admin store mirror.
        useAdminUserStore.getState().revokeSubmission(sessionUser.id);
      }
      // BE returns the new status — use it directly. Should always be
      // "incomplete" after a successful withdraw; defensive narrow anyway.
      const next =
        response.status === "approved" ||
        response.status === "pending" ||
        response.status === "incomplete"
          ? response.status
          : "incomplete";
      setStatus(next);
      showToast({
        kind: "info",
        message: "Application withdrawn.",
      });
    } catch (err) {
      console.error("[photographer/settings] withdrawVerification failed", err);
      showToast({
        kind: "error",
        message: "Couldn't withdraw — check your connection and try again.",
      });
    }
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
          as we&apos;re done. Edits below stay live — the admin always sees your
          latest changes.
        </p>
        <div className="mt-5">
          <button
            type="button"
            onClick={handleWithdraw}
            className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
          >
            Withdraw application
          </button>
        </div>
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
        Profile picture, cover, brand, watermark, public URL, region, at least
        one social link, and at least one payout account are required so we can
        verify you and send your sales. Bio is optional.
      </p>
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!complete || submitting}
        className="mt-5 inline-flex items-center gap-2 font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
      >
        {!complete
          ? "Fill the required fields"
          : submitting
            ? "Submitting…"
            : "Submit for review"}
        {complete && !submitting && <span aria-hidden="true">→</span>}
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
  const { confirm } = useConfirmation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      // Local center-crop stages an instant preview; backend runs the same
      // pipeline server-side and returns the canonical avatarUrl on auth.
      const dataUrl = await squareCropToDataUrl(file, AVATAR_SIZE_PX);
      setAvatar({ dataUrl, uploadedAt: new Date().toISOString() });
      fireSettings("uploadAvatar", uploadAvatar(file));
      showToast({ kind: "success", message: "Profile picture updated." });
    } catch (err) {
      setError(imageErrorCopy(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    const ok = await confirm({
      title: "Remove profile picture?",
      message:
        "Your initials will show next to your name until you upload a new one.",
      confirmLabel: "Remove",
    });
    if (!ok) return;
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
              ref={inputRef}
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handlePick}
              disabled={busy}
              aria-describedby={error ? "picture-error" : undefined}
              aria-invalid={error ? true : undefined}
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
      <FieldError message={error} id="picture-error" />
      {error && !busy && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors mt-2"
        >
          Try again
        </button>
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
  const { confirm } = useConfirmation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    // Inspect dimensions BEFORE setting busy — the soft confirmation may
    // bounce the upload, and we don't want the button stuck in Processing
    // while the user is reading the dialog. If decode fails here, surface
    // the same error copy as the encode path.
    let aspectRatio: number;
    try {
      const dims = await inspectImageDimensions(file);
      aspectRatio = dims.aspectRatio;
    } catch (err) {
      setError(imageErrorCopy(err));
      return;
    }

    if (aspectRatio < COVER_MIN_ASPECT) {
      const ok = await confirm({
        title: "This crops to a wide banner.",
        message:
          "Your image is taller than wide. Only the middle horizontal strip will show. Use it anyway?",
        confirmLabel: "Use it",
      });
      if (!ok) {
        setError(null);
        return;
      }
    }

    setBusy(true);
    setError(null);
    try {
      // Local downscale keeps the optimistic preview snappy; backend re-runs
      // the same downscale + JPEG re-encode and returns the canonical URL.
      const dataUrl = await fitToDataUrl(file, COVER_MAX_PX, 0.82);
      setCover({ dataUrl, uploadedAt: new Date().toISOString() });
      fireSettings("postCover", postCover(file));
      showToast({ kind: "success", message: "Cover updated." });
    } catch (err) {
      setError(imageErrorCopy(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    const ok = await confirm({
      title: "Remove cover?",
      message:
        "Your portfolio will fall back to the default banner until you upload a new one.",
      confirmLabel: "Remove",
    });
    if (!ok) return;
    setCover(null);
    setError(null);
    showToast({ kind: "success", message: "Cover removed." });
  }

  return (
    <div>
      <SubHeading label="Cover banner" caption="16:5 · 1920px wide · 8 MB" />
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
            ref={inputRef}
            type="file"
            accept={ACCEPTED_IMAGE_MIME.join(",")}
            onChange={handlePick}
            disabled={busy}
            aria-describedby={error ? "cover-error" : undefined}
            aria-invalid={error ? true : undefined}
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
      <FieldError message={error} id="cover-error" />
      {error && !busy && (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors mt-2"
        >
          Try again
        </button>
      )}
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

  // Track touch so an unfilled name on first render doesn't shout at the
  // user — the error only appears after they've interacted (blur or first
  // edit) and the field is still invalid.
  const [nameTouched, setNameTouched] = useState(false);

  // Silent commit — auto-save (debounce-pause) writes to the store without
  // toasting so rapid editing doesn't flood the user with "Saved." pings.
  // Explicit Save click runs flush() then surfaces a single confirmation
  // toast in handleSubmit below. Live mode debounce-fires PUT /me/photographer/brand.
  const saveBrand = useCallback(
    async (draft: { name: string; bio: string }) => {
      const trimmed = draft.name.trim();
      setBrandName(trimmed);
      setBio(draft.bio);
      fireSettings(
        "putBrand",
        putBrand({ brandName: trimmed, brandColor, bio: draft.bio }),
      );
    },
    [setBrandName, setBio, brandColor],
  );

  const { save, flush, isSaving } = useDebouncedSave(saveBrand);

  const dirty = nameDraft.trim() !== brandName || bioDraft !== bio;
  const nameHasNewline = /[\r\n]/.test(nameDraft);
  const validName =
    nameDraft.trim().length > 0 &&
    nameDraft.trim().length <= BRAND_NAME_MAX &&
    !nameHasNewline;
  const validBio = bioDraft.length <= BIO_MAX;
  const nameError =
    nameTouched && nameDraft.trim().length === 0
      ? "Display name is required."
      : nameTouched && nameHasNewline
        ? "Display name can't contain a line break."
        : null;

  // Schedule a debounced auto-save whenever the draft is dirty AND valid.
  // Invalid drafts (empty / too long / newline) hold until the user fixes
  // them; the explicit Save button still runs through flush().
  useEffect(() => {
    if (dirty && validName && validBio) {
      save({ name: nameDraft, bio: bioDraft });
    }
  }, [dirty, validName, validBio, nameDraft, bioDraft, save]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty || !validName || !validBio) return;
    await flush();
    showToast({ kind: "success", message: "Brand saved." });
  }

  const saving = isSaving;

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
            onChange={(e) => {
              setNameDraft(e.target.value);
              if (!nameTouched) setNameTouched(true);
            }}
            onBlur={() => setNameTouched(true)}
            placeholder="Cebu Coastline Photo"
            maxLength={BRAND_NAME_MAX}
            required
            aria-describedby={
              nameError ? "brand-name-error" : "brand-name-counter"
            }
            aria-invalid={nameError ? true : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-4 text-lg text-ink placeholder:text-slate-soft transition-colors"
          />
          <CharCounter
            id="brand-name-counter"
            current={nameDraft.trim().length}
            max={BRAND_NAME_MAX}
          />
          <FieldError
            message={nameError}
            id="brand-name-error"
            density="tight"
          />
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
          <CharCounter current={bioDraft.length} max={BIO_MAX} />
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
  const { confirm } = useConfirmation();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    // Replacement confirmation — unique to watermark because it retroactively
    // affects how runners perceive the brand on the gallery (already-stamped
    // photos won't re-stamp, but new uploads will use the new mark; warn so
    // the photographer doesn't accidentally split their gallery's look).
    if (watermark) {
      const ok = await confirm({
        title: "Replace watermark?",
        message:
          "New uploads will use the new watermark. Photos already stamped won't change.",
        confirmLabel: "Replace",
      });
      if (!ok) {
        setError(null);
        return;
      }
    }

    setBusy(true);
    setError(null);
    try {
      // Local downscale stages an instant preview; backend re-runs the same
      // pipeline server-side so the watermark composites at upload time.
      // PNG preserves transparency for the preview; JPEG/WebP fall through to
      // fitToDataUrl which encodes JPEG (no alpha channel needed).
      const dataUrl =
        file.type === "image/png"
          ? await fitToPngDataUrl(file, WATERMARK_MAX_PX)
          : await fitToDataUrl(file, WATERMARK_MAX_PX);
      setWatermark({ dataUrl, uploadedAt: new Date().toISOString() });
      fireSettings("postWatermark", postWatermark(file));
      showToast({ kind: "success", message: "Watermark updated." });
    } catch (err) {
      setError(imageErrorCopy(err));
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    const ok = await confirm({
      title: "Remove watermark?",
      message:
        "New uploads will go out unwatermarked until you replace it. Existing photos already stamped won't change.",
      confirmLabel: "Remove",
    });
    if (!ok) return;
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
          how they find you. A transparent PNG overlays cleanly, but a logo
          with a solid background works too.
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
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={handlePick}
              disabled={busy}
              aria-describedby={error ? "watermark-error" : undefined}
              aria-invalid={error ? true : undefined}
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
            PNG, JPEG, or WebP · downscaled to 600px wide · 8 MB max.
          </p>
        </div>

        <FieldError message={error} id="watermark-error" />
        {error && !busy && (
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors mt-2"
          >
            Try again
          </button>
        )}
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

  // Silent commit — same pattern as BrandSubsection. Auto-save fires on
  // debounce-pause without toasting; explicit Save click runs flush() then
  // surfaces a single confirmation toast. Live mode debounce-fires
  // PUT /me/photographer/handle (backend returns 409 on collision; we log,
  // store stays optimistic, photographer sees error on next refresh).
  const saveHandle = useCallback(
    async (next: string) => {
      setHandle(next);
      fireSettings("putHandle", putHandle(next));
    },
    [setHandle],
  );

  const { save, flush, isSaving } = useDebouncedSave(saveHandle);

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

  // Schedule a debounced auto-save when the draft is dirty AND valid. Invalid
  // drafts hold (FieldError surfaces them); explicit Save button still runs
  // through flush().
  useEffect(() => {
    if (dirty && valid) {
      save(draft);
    }
  }, [dirty, valid, draft, save]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!dirty || !valid) return;
    await flush();
    showToast({ kind: "success", message: "Public URL updated." });
  }

  const saving = isSaving;

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
                // Lowercase only (canonical form for the URL — this is
                // normalization, not auto-correct of invalid characters).
                // Anything else the user types — spaces, emoji, > 32 chars —
                // stays in the field and surfaces via <FieldError> below so
                // the user can see what they typed and fix it themselves.
                setDraft(e.target.value.toLowerCase())
              }
              placeholder="your-handle"
              autoComplete="off"
              spellCheck={false}
              required
              aria-describedby={
                validation && draft.length > 0
                  ? "handle-error"
                  : "handle-hint"
              }
              aria-invalid={
                validation && draft.length > 0 ? true : undefined
              }
              className="flex-1 min-w-0 bg-transparent focus:outline-none py-4 text-lg font-mono text-ink placeholder:text-slate-soft"
            />
          </div>
          {draft.length === 0 ? (
            <p
              id="handle-hint"
              className="font-sans text-sm text-slate-soft"
            >
              Lowercase letters, numbers, and dashes. 3–32 characters.
            </p>
          ) : (
            <FieldError
              message={validation}
              id="handle-error"
              density="tight"
            />
          )}
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

  // A persisted region/province code can fail to resolve when the backend
  // renames or removes one (Phase B+ — the mock list never changes). Surface
  // the gap so the user knows to re-pick instead of silently rendering
  // "Pick a region" as if nothing was set.
  const staleRegion = !!region && !selectedRegion;
  const staleProvince =
    !!region && !!selectedRegion && !!region.provinceCode && !selectedProvince;

  function handlePickRegion(regionCode: string) {
    if (region?.regionCode === regionCode) return;
    // Reset province when region changes — the user must re-pick.
    const next = getRegion(regionCode);
    if (!next) return;
    const updated = { regionCode, provinceCode: "" };
    setRegion(updated);
    fireSettings("putRegion", putRegion(updated));
  }

  function handlePickProvince(provinceCode: string) {
    if (!region) return;
    if (region.provinceCode === provinceCode) return;
    const updated = { regionCode: region.regionCode, provinceCode };
    setRegion(updated);
    fireSettings("putRegion", putRegion(updated));
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
                <span
                  aria-describedby={staleRegion ? "region-error" : undefined}
                  aria-invalid={staleRegion ? true : undefined}
                  className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left transition-colors hover:border-slate"
                >
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
            <FieldError
              message={
                staleRegion
                  ? "That region is no longer available. Pick a new one."
                  : null
              }
              id="region-error"
              density="tight"
            />
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
                  <span
                    aria-describedby={
                      staleProvince ? "province-error" : undefined
                    }
                    aria-invalid={staleProvince ? true : undefined}
                    className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left transition-colors hover:border-slate"
                  >
                    <span
                      className={cn(
                        "truncate text-lg",
                        selectedProvince ? "text-ink" : "text-slate-soft",
                      )}
                    >
                      {selectedProvince?.name ??
                        (selectedRegion.provinces.length === 0
                          ? "No provinces in this region"
                          : "Pick a province")}
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
            <FieldError
              message={
                staleProvince
                  ? "That province is no longer available. Pick a new one."
                  : null
              }
              id="province-error"
              density="tight"
            />
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
  const { confirm } = useConfirmation();

  function handleAdd(platform: SocialPlatform) {
    addSocial(platform, "");
    showToast({
      kind: "info",
      message: `${SOCIAL_PLATFORM_LABEL[platform]} row added — drop your link in.`,
    });
  }

  async function handleRemove(id: string, platform: SocialPlatform) {
    const label = SOCIAL_PLATFORM_LABEL[platform];
    const ok = await confirm({
      title: `Remove ${label} link?`,
      message:
        "Runners use these to verify you're real. You can add it back anytime.",
      confirmLabel: "Remove",
    });
    if (!ok) return;
    removeSocial(id);
    showToast({
      kind: "success",
      message: `${label} link removed.`,
    });
  }

  const filledCount = socials.filter((s) => s.url.trim().length > 0).length;

  // Compute the set of normalized URLs that appear in more than one row, so
  // each affected row can flag itself. Skips empty values — those are caught
  // by the empty-row check, not the duplicate check.
  const duplicateUrls = (() => {
    const counts = new Map<string, number>();
    for (const link of socials) {
      const trimmed = link.url.trim();
      if (trimmed.length === 0) continue;
      const key = normalizeSocialUrl(trimmed);
      counts.set(key, (counts.get(key) ?? 0) + 1);
    }
    const dupes = new Set<string>();
    for (const [key, count] of counts) {
      if (count > 1) dupes.add(key);
    }
    return dupes;
  })();

  const invalidCount = socials.filter((link) =>
    validateSocialUrl(link.platform, link.url, { duplicateUrls }) !== null
  ).length;

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
                duplicateUrls={duplicateUrls}
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
          <Kicker
            as="p"
            tone="soft"
            tnum
            className={cn(invalidCount > 0 && "text-warning")}
          >
            {filledCount} of {socials.length} filled in
            {invalidCount > 0 && ` · ${invalidCount} need${invalidCount === 1 ? "s" : ""} attention`}
          </Kicker>
        )}
      </div>
    </Slab>
  );
}

interface SocialRowProps {
  link: SocialLink;
  duplicateUrls: ReadonlySet<string>;
  onChange: (url: string) => void;
  onRemove: () => void;
}

function SocialRow({
  link,
  duplicateUrls,
  onChange,
  onRemove,
}: SocialRowProps) {
  // Mirrors the brand-name pattern: an empty row that the user just added
  // shouldn't shout "Add a URL or remove this row." until they've engaged
  // with the input. First keystroke or first blur flips this on.
  const [touched, setTouched] = useState(link.url.trim().length > 0);

  const placeholder: Record<SocialPlatform, string> = {
    facebook: "https://facebook.com/your-page",
    instagram: "https://instagram.com/your-handle",
    tiktok: "https://tiktok.com/@your-handle",
    x: "https://x.com/your-handle",
    youtube: "https://youtube.com/@your-channel",
    website: "https://your-site.com",
  };

  const validation = validateSocialUrl(link.platform, link.url, {
    duplicateUrls,
  });
  const errorMessage = touched ? validation : null;
  const errorId = `social-error-${link.id}`;

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
          onChange={(e) => {
            onChange(e.target.value);
            if (!touched) setTouched(true);
          }}
          onBlur={() => setTouched(true)}
          placeholder={placeholder[link.platform]}
          maxLength={SOCIAL_URL_MAX}
          aria-describedby={errorMessage ? errorId : undefined}
          aria-invalid={errorMessage ? true : undefined}
          className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          inputMode="url"
          autoComplete="off"
          spellCheck={false}
        />
        <FieldError message={errorMessage} id={errorId} density="tight" />
      </div>
    </li>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Slab 06 — Payout accounts
// ─────────────────────────────────────────────────────────────────────────

// Validation contract:
//   - Empty input is null (required-ness is the form's job, not this helper's).
//   - Overflow checks fire BEFORE shape checks so a 30-digit paste reads as
//     "Mobile number is at most 11 digits." instead of "PH mobile numbers
//     start with 09." (the displayed value gets format-truncated, but the
//     stored draft holds the full paste — surface the overflow explicitly so
//     the user understands why their typed value differs from what shows).
function validatePayoutNumber(method: PayoutMethod, raw: string): string | null {
  const d = raw.replace(/\D/g, "");
  if (method === "gcash" || method === "maya") {
    if (d.length === 0) return null;
    if (d.length > 11) return "Mobile number is at most 11 digits.";
    if (d.length !== 11) return "Mobile number is 11 digits.";
    if (!d.startsWith("09")) return "PH mobile numbers start with 09.";
    return null;
  }
  if (d.length === 0) return null;
  if (d.length > 16) return "Account number is at most 16 digits.";
  if (d.length < 10) return "Account number is at least 10 digits.";
  return null;
}

function PayoutSlab() {
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  const addPayout = usePhotographerSettingsStore((s) => s.addPayout);
  const updatePayout = usePhotographerSettingsStore((s) => s.updatePayout);
  const setPrimaryPayout = usePhotographerSettingsStore(
    (s) => s.setPrimaryPayout,
  );
  const removePayout = usePhotographerSettingsStore((s) => s.removePayout);
  const { showToast } = useToast();
  const { confirm } = useConfirmation();

  const [draftMethod, setDraftMethod] = useState<PayoutMethod | null>(null);
  // Only one row may be in edit mode at a time — opening Edit on another row
  // collapses any in-flight edit. Adding a new account also closes any open
  // edit to avoid two forms competing for the photographer's attention.
  const [editingId, setEditingId] = useState<string | null>(null);

  function handlePickMethod(method: PayoutMethod) {
    setEditingId(null);
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

  function handleStartEdit(id: string) {
    setDraftMethod(null);
    setEditingId(id);
  }

  function handleUpdate(
    id: string,
    method: PayoutMethod,
    input: {
      accountNumber: string;
      accountName: string;
      qr: { dataUrl: string; uploadedAt: string } | null;
    },
  ) {
    updatePayout(id, input);
    setEditingId(null);
    showToast({
      kind: "success",
      message: `${PAYOUT_METHOD_LABEL[method]} account updated.`,
    });
  }

  function handleMakePrimary(id: string, method: PayoutMethod) {
    setPrimaryPayout(id);
    showToast({
      kind: "success",
      message: `${PAYOUT_METHOD_LABEL[method]} is now your primary payout.`,
    });
  }

  async function handleRemove(id: string, method: PayoutMethod) {
    const label = PAYOUT_METHOD_LABEL[method];
    const isOnlyPayout = payouts.length === 1;
    const account = payouts.find((p) => p.id === id);
    const isPrimary = account?.isPrimary ?? false;
    const ok = await confirm({
      title: `Remove ${label} account?`,
      message: isOnlyPayout
        ? "This is your only payout account. Without one, sales can't be paid out until you add a new account."
        : isPrimary
          ? "This is your primary account. We'll promote one of your other accounts to primary automatically."
          : "Future sales won't pay out to this account anymore.",
      confirmLabel: "Remove",
      danger: true,
    });
    if (!ok) return;
    removePayout(id);
    showToast({
      kind: "success",
      message: `${label} account removed.`,
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
              {payouts.map((account) =>
                editingId === account.id ? (
                  <li key={account.id}>
                    <PayoutForm
                      method={account.method}
                      initial={{
                        accountName: account.accountName,
                        accountNumber: account.accountNumber,
                        qr: account.qr,
                      }}
                      submitLabel="Update account"
                      titleLabel={`Edit ${PAYOUT_METHOD_LABEL[account.method]} account`}
                      onCancel={() => setEditingId(null)}
                      onSave={(input) =>
                        handleUpdate(account.id, account.method, input)
                      }
                    />
                  </li>
                ) : (
                  <PayoutCard
                    key={account.id}
                    account={account}
                    onEdit={() => handleStartEdit(account.id)}
                    onMakePrimary={() =>
                      handleMakePrimary(account.id, account.method)
                    }
                    onRemove={() => handleRemove(account.id, account.method)}
                  />
                ),
              )}
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
  /** Pre-fill values for edit mode. Omitted for add mode. */
  initial?: {
    accountName: string;
    accountNumber: string;
    qr: { dataUrl: string; uploadedAt: string } | null;
  };
  /** Defaults to "Save account" (add mode). Pass "Update account" for edit. */
  submitLabel?: string;
  /** Header label inside the form card. Defaults to "New {method} account". */
  titleLabel?: string;
  onCancel: () => void;
  onSave: (input: {
    method: PayoutMethod;
    accountNumber: string;
    accountName: string;
    qr: { dataUrl: string; uploadedAt: string } | null;
  }) => void;
}

function PayoutForm({
  method,
  initial,
  submitLabel = "Save account",
  titleLabel,
  onCancel,
  onSave,
}: PayoutFormProps) {
  const { showToast } = useToast();
  const isEdit = !!initial;
  const [accountName, setAccountName] = useState(initial?.accountName ?? "");
  const [nameTouched, setNameTouched] = useState(false);
  const [accountNumber, setAccountNumber] = useState(
    initial?.accountNumber ?? "",
  );
  const [qr, setQr] = useState<{ dataUrl: string; uploadedAt: string } | null>(
    initial?.qr ?? null,
  );
  const [busy, setBusy] = useState(false);
  const [qrError, setQrError] = useState<string | null>(null);
  const qrInputRef = useRef<HTMLInputElement>(null);

  const numberError = validatePayoutNumber(method, accountNumber);
  const validNumber =
    accountNumber.replace(/\D/g, "").length > 0 && numberError === null;
  const validName = accountName.trim().length > 0;
  // TODO(backend): per the plan, real account-name match validation lives on
  // the server — `POST /me/photographer/payouts/verify` returns whether the
  // typed name matches the wallet record. Until then the inline rule is just
  // a non-empty check; the helper text below sets the expectation.
  const nameError =
    nameTouched && accountName.trim().length === 0
      ? "Account name is required."
      : null;
  const canSave = validNumber && validName;

  const numberLabel = method === "gotyme" ? "Account number" : "Mobile number";
  const numberPlaceholder =
    method === "gotyme" ? "1234 5678 9012" : "0917 555 0101";
  const numberHelper =
    method === "gotyme"
      ? "Your GoTyme account number, digits only."
      : "11-digit mobile number registered to this wallet.";

  async function handleQrPick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setQrError(validationError);
      return;
    }

    setBusy(true);
    setQrError(null);
    try {
      // TODO(backend): swap for `api.post("/me/photographer/payouts/{id}/qr", formData)`
      // — server stores the image privately; only finance staff can read it.
      const dataUrl = await fitToPngDataUrl(file, QR_MAX_PX);
      setQr({ dataUrl, uploadedAt: new Date().toISOString() });
    } catch (err) {
      setQrError(imageErrorCopy(err));
    } finally {
      setBusy(false);
    }
  }

  function handleQrRetry() {
    setQrError(null);
    qrInputRef.current?.click();
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
          {titleLabel ?? `New ${PAYOUT_METHOD_LABEL[method]} account`}
        </p>
        <button
          type="button"
          onClick={() => {
            onCancel();
            showToast({
              kind: "info",
              message: isEdit ? "Edit discarded." : "New account discarded.",
            });
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
            onChange={(e) => {
              setAccountName(e.target.value);
              if (!nameTouched) setNameTouched(true);
            }}
            onBlur={() => setNameTouched(true)}
            placeholder="Juan Dela Cruz"
            maxLength={ACCOUNT_NAME_MAX}
            required
            autoComplete="off"
            aria-describedby={
              nameError
                ? `payout-name-error-${method}`
                : `payout-name-counter-${method}`
            }
            aria-invalid={nameError ? true : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors"
          />
          <CharCounter
            id={`payout-name-counter-${method}`}
            current={accountName.length}
            max={ACCOUNT_NAME_MAX}
          />
          {nameError ? (
            <FieldError
              message={nameError}
              id={`payout-name-error-${method}`}
              density="tight"
            />
          ) : (
            <p className="font-sans text-sm text-slate-soft">
              Must match the name on your {PAYOUT_METHOD_LABEL[method]} account.
            </p>
          )}
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
            aria-describedby={
              numberError
                ? `payout-number-error-${method}`
                : `payout-number-hint-${method}`
            }
            aria-invalid={numberError ? true : undefined}
            className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base font-mono tnum text-ink placeholder:text-slate-soft transition-colors"
          />
          {numberError ? (
            <FieldError
              message={numberError}
              id={`payout-number-error-${method}`}
              density="tight"
            />
          ) : (
            <p
              id={`payout-number-hint-${method}`}
              className="font-sans text-sm text-slate-soft"
            >
              {numberHelper}
            </p>
          )}
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
                ref={qrInputRef}
                type="file"
                accept={ACCEPTED_IMAGE_MIME.join(",")}
                onChange={handleQrPick}
                disabled={busy}
                aria-describedby={qrError ? `payout-qr-error-${method}` : undefined}
                aria-invalid={qrError ? true : undefined}
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

        <FieldError
          message={qrError}
          id={`payout-qr-error-${method}`}
          density="tight"
        />
        {qrError && !busy && (
          <button
            type="button"
            onClick={handleQrRetry}
            className="self-start font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
          >
            Try again
          </button>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 pt-2">
        <button
          type="submit"
          disabled={!canSave}
          className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
        >
          {submitLabel}
          <span aria-hidden="true">→</span>
        </button>
      </div>
    </form>
  );
}

interface PayoutCardProps {
  account: PayoutAccount;
  onEdit: () => void;
  onMakePrimary: () => void;
  onRemove: () => void;
}

function PayoutCard({
  account,
  onEdit,
  onMakePrimary,
  onRemove,
}: PayoutCardProps) {
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
          onClick={onEdit}
          className="font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
        >
          Edit
        </button>
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
