"use client";

import { useCallback, useEffect } from "react";
import {
  fetchVerificationStatus,
  type VerificationSubmitResponse,
} from "@/lib/api-photographer-settings";
import { useAuthStore } from "@/store/auth-store";
import {
  usePhotographerSettingsStore,
  type VerificationStatus,
} from "@/store/photographer-settings-store";

// Phase 4 — cross-session verification visibility. Pull the photographer's
// current status + missing[] from the BE on three triggers:
//   1. On mount of the host page.
//   2. When the tab/window regains focus (the photographer was looking at
//      something else — including the admin view in another tab).
//   3. Every 30s while local status is "pending" (waiting on an admin decision).
//
// Once the BE confirms "approved", interval polling stops. Focus + mount sync
// still fire so a future admin reset/suspend is surfaced on the next visit.

const POLL_INTERVAL_MS = 30_000;

function normaliseStatus(raw: string): VerificationStatus {
  // BE returns lowercase wire form; REJECTED is collapsed to incomplete by the
  // admin reject action, so we should never see it — fall back defensively.
  return raw === "approved" || raw === "pending" || raw === "incomplete"
    ? raw
    : "incomplete";
}

// Fold a /me/photographer/verification response into the store. The single
// place status + missing[] + suspension are written, so the poll, the
// focus/mount sync, and the post-Save refresh can never diverge. missing[] is
// the completeness source of truth — no client-side recomputation.
export function applyVerification(response: VerificationSubmitResponse): void {
  const state = usePhotographerSettingsStore.getState();

  const next = normaliseStatus(response.status);
  if (next !== state.verificationStatus) state.setVerificationStatus(next);

  const nextSuspendedAt = response.suspendedAt ?? null;
  const nextReason = response.suspensionReason ?? null;
  if (
    nextSuspendedAt !== state.suspendedAt ||
    nextReason !== state.suspensionReason
  ) {
    state.setSuspension(nextSuspendedAt, nextReason);
  }

  const nextMissing = response.missing ?? [];
  const nextCodes = response.missingCodes ?? [];
  if (
    state.verificationMissing === null ||
    state.verificationMissing.join("|") !== nextMissing.join("|")
  ) {
    state.setVerificationMissing(nextMissing, nextCodes);
  }
}

// Fetch + apply once, guarded against overlap. Callable outside React — the
// settings page calls it right after a successful Save so completeness updates
// immediately instead of waiting for the next focus/poll.
let inFlight = false;
export async function syncVerificationNow(): Promise<void> {
  if (inFlight) return;
  inFlight = true;
  try {
    applyVerification(await fetchVerificationStatus());
  } catch (err) {
    // Network blip or 401-after-logout. Stay on cached state and try again on
    // the next trigger — no toast, this is a background sync.
    console.warn("[photographer/verification] sync failed", err);
  } finally {
    inFlight = false;
  }
}

// Safety: only runs while the user is logged in AND has role PHOTOGRAPHER.
// The BE returns "incomplete" for users with no settings row, so this is also
// safe for fresh accounts.
export function usePhotographerVerificationSync(): void {
  const sessionUser = useAuthStore((s) => s.user);
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const isPhotographer = sessionUser?.role === "PHOTOGRAPHER";

  const syncOnce = useCallback(() => {
    void syncVerificationNow();
  }, []);

  // Mount + focus-driven sync. visibilitychange + focus both fire on an
  // ordinary tab return, ms apart — the two listeners exist to also cover
  // alt-tab (focus without a visibility flip); syncVerificationNow's in-flight
  // guard joins the pending call instead of issuing a second GET.
  useEffect(() => {
    if (!isPhotographer) return;
    syncOnce();

    function onVisible() {
      if (!document.hidden) syncOnce();
    }
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("focus", onVisible);
    return () => {
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("focus", onVisible);
    };
  }, [isPhotographer, syncOnce]);

  // Interval poll only while pending — no point pulling once approved.
  useEffect(() => {
    if (!isPhotographer || status !== "pending") return;
    const interval = setInterval(syncOnce, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [isPhotographer, status, syncOnce]);
}
