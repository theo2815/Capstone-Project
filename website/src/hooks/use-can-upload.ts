"use client";

import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";
import { useUserMediaStore } from "@/store/user-media-store";

export type UploadGateState =
  | { kind: "ok" }
  | { kind: "suspended"; reason: string | null }
  | { kind: "incomplete"; missingLabel: string }
  | { kind: "pending" };

// Single source of truth for "can this photographer upload right now?". Used
// by upload pages to gate the form, and by the dashboard overview / event
// pages to show the verification banner.
//
// Priority order: suspended > incomplete > pending > ok. Suspended is the
// hardest block — even an approved photographer drops here the moment admin
// flips suspendedAt. Source is photographer-verification-sync polling
// /me/photographer/verification (mount + focus + 30s while pending).
export function useCanUpload(): UploadGateState {
  const avatar = useUserMediaStore((s) => s.avatar);
  const cover = usePhotographerSettingsStore((s) => s.cover);
  const brandName = usePhotographerSettingsStore((s) => s.brandName);
  const watermark = usePhotographerSettingsStore((s) => s.watermark);
  const handle = usePhotographerSettingsStore((s) => s.handle);
  const region = usePhotographerSettingsStore((s) => s.region);
  const socials = usePhotographerSettingsStore((s) => s.socials);
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const suspendedAt = usePhotographerSettingsStore((s) => s.suspendedAt);
  const suspensionReason = usePhotographerSettingsStore(
    (s) => s.suspensionReason,
  );

  if (suspendedAt) {
    return { kind: "suspended", reason: suspensionReason };
  }

  if (status === "approved") {
    return { kind: "ok" };
  }

  // Incomplete state — surface which field is still missing so the banner
  // can name it. Order matches the Settings page slab order.
  if (avatar === null)
    return { kind: "incomplete", missingLabel: "profile picture" };
  if (cover === null) return { kind: "incomplete", missingLabel: "cover banner" };
  if (brandName.trim().length === 0)
    return { kind: "incomplete", missingLabel: "brand name" };
  if (watermark === null)
    return { kind: "incomplete", missingLabel: "watermark" };
  if (handle.trim().length === 0)
    return { kind: "incomplete", missingLabel: "public URL handle" };
  if (region === null)
    return { kind: "incomplete", missingLabel: "region" };
  if (!socials.some((s) => s.url.trim().length > 0))
    return { kind: "incomplete", missingLabel: "social link" };
  if (!payouts.some((p) => p.isPrimary))
    return { kind: "incomplete", missingLabel: "payout account" };

  // All fields filled but verification still pending review.
  return { kind: "pending" };
}
