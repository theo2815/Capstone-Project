"use client";

import { useCallback, useEffect, useRef } from "react";
import { fetchVerificationStatus } from "@/lib/api-photographer-settings";
import { useAuthStore } from "@/store/auth-store";
import {
  usePhotographerSettingsStore,
  type VerificationStatus,
} from "@/store/photographer-settings-store";

// Phase 4 — cross-session verification visibility. Pull the photographer's
// current status from the BE on three triggers:
//   1. On mount of the host page.
//   2. When the tab/window regains focus (the photographer was looking at
//      something else — including the admin view in another tab).
//   3. Every 30s while local status is "pending" (the photographer is
//      sitting on /dashboard/settings waiting for the admin to decide).
//
// Once the BE confirms "approved", interval polling stops — there's no
// further state to wait for. Focus + mount sync still fire so a future
// admin reset/suspend is still surfaced on the next page visit.
//
// Safety: only runs while the user is logged in AND has role PHOTOGRAPHER.
// Anything else short-circuits (no network call). The BE returns "incomplete"
// for users with no settings row, so this is also safe for fresh accounts.

const POLL_INTERVAL_MS = 30_000;

export function usePhotographerVerificationSync(): void {
  const sessionUser = useAuthStore((s) => s.user);
  const status = usePhotographerSettingsStore((s) => s.verificationStatus);
  const setStatus = usePhotographerSettingsStore(
    (s) => s.setVerificationStatus,
  );
  const setSuspension = usePhotographerSettingsStore((s) => s.setSuspension);

  const isPhotographer = sessionUser?.role === "PHOTOGRAPHER";

  // visibilitychange + focus both fire on an ordinary tab return, ms apart —
  // the two listeners exist to also cover alt-tab (focus without a visibility
  // flip). Join the in-flight sync instead of issuing a second GET.
  const inFlightRef = useRef(false);

  const syncOnce = useCallback(async () => {
    if (inFlightRef.current) return;
    inFlightRef.current = true;
    try {
      const response = await fetchVerificationStatus();
      const state = usePhotographerSettingsStore.getState();
      // BE returns lowercase wire form; the FE type accepts "approved" |
      // "pending" | "incomplete". REJECTED is collapsed to incomplete by the
      // admin reject action, so we should never see it here — but if we do,
      // fall back to incomplete defensively.
      const next: VerificationStatus =
        response.status === "approved" ||
        response.status === "pending" ||
        response.status === "incomplete"
          ? response.status
          : "incomplete";
      if (next !== state.verificationStatus) {
        setStatus(next);
      }
      const nextSuspendedAt = response.suspendedAt ?? null;
      const nextReason = response.suspensionReason ?? null;
      if (
        nextSuspendedAt !== state.suspendedAt ||
        nextReason !== state.suspensionReason
      ) {
        setSuspension(nextSuspendedAt, nextReason);
      }
    } catch (err) {
      // Network blip or 401-after-logout. Stay on cached state and try
      // again on the next trigger — no toast, this is a background sync.
      console.warn("[photographer/verification] sync failed", err);
    } finally {
      inFlightRef.current = false;
    }
  }, [setStatus, setSuspension]);

  // Mount + focus-driven sync. Effect runs once per photographer session.
  useEffect(() => {
    if (!isPhotographer) return;
    void syncOnce();

    function onVisible() {
      if (!document.hidden) void syncOnce();
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
