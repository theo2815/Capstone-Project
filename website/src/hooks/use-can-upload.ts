"use client";

import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

export type UploadGateState =
  | { kind: "ok" }
  | { kind: "suspended"; reason: string | null }
  | { kind: "incomplete"; missing: string[]; missingLabel: string }
  | { kind: "pending" };

// Single source of truth for "can this photographer act right now?". Derived
// entirely from the backend verification poll — photographer-verification-sync
// writes status + missing[] + suspension into the store. The client NEVER
// recomputes completeness from individual field stores (that drift was the
// "everything uploaded but still says incomplete" bug). Consumed by upload
// pages, the dashboard verification banner, and the dashboard railguard.
//
// Priority: suspended > incomplete > pending > ok. Suspended is the hardest
// block — even an approved photographer drops here the moment admin flips
// suspendedAt.
export function useCanUpload(): UploadGateState {
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const missing = usePhotographerSettingsStore((s) => s.verificationMissing);
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

  // missing === null means the verification poll hasn't landed this session
  // yet. Fall back to status so we never flash "ok"/"ready" for an unverified
  // photographer: pending stays pending, anything else guides to settings.
  if (missing === null) {
    return status === "pending"
      ? { kind: "pending" }
      : { kind: "incomplete", missing: [], missingLabel: "your settings" };
  }

  if (missing.length > 0) {
    return { kind: "incomplete", missing, missingLabel: missing[0] };
  }

  // Every required field is filled but the account isn't approved yet —
  // awaiting admin review (the settings page's own panel carries the submit
  // CTA when the photographer hasn't submitted).
  return { kind: "pending" };
}
