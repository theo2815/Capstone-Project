"use client";

import { useEffect } from "react";
import type { ReactNode } from "react";
import { PhotographerNotReadyNotice } from "@/components/dashboard/photographer-not-ready-notice";
import { useCanUpload } from "@/hooks/use-can-upload";
import { syncVerificationNow } from "@/lib/photographer-verification-sync";
import { usePhotographerSettingsStore } from "@/store/photographer-settings-store";

// Centralized dashboard railguard. Wrap any feature that requires a complete,
// admin-approved photographer (create event, and future create/sell actions):
// it renders `children` only when the backend-driven gate is "ok", otherwise a
// friendly notice that names what's missing and routes to settings. The backend
// (PhotographerAccessGuard) enforces the same rule — this is the guiding UX so
// the photographer never hits a raw error mid-flow. New restricted features
// inherit the behavior by wrapping in this instead of re-deriving a check.
export function RequirePhotographerReady({
  children,
}: {
  children: ReactNode;
}) {
  // Self-sync when the store has no backend answer yet. DashboardShell mounts
  // the verification poll for most routes, but "focused" routes escape the
  // shell (e.g. /dashboard/events/new, matched by the events/[id] pattern in
  // dashboard/layout.tsx) — there the gate would otherwise read stale/seed
  // state and wrongly block a ready photographer. syncVerificationNow's
  // in-flight guard dedupes against the shell's poll on ordinary routes.
  const unsynced = usePhotographerSettingsStore(
    (s) => s.verificationMissing === null,
  );
  useEffect(() => {
    if (unsynced) void syncVerificationNow();
  }, [unsynced]);

  const gate = useCanUpload();
  if (gate.kind === "ok") return <>{children}</>;
  return <PhotographerNotReadyNotice gate={gate} />;
}
