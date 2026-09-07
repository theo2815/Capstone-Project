"use client";

import type { ReactNode } from "react";
import { PhotographerNotReadyNotice } from "@/components/dashboard/photographer-not-ready-notice";
import { useCanUpload } from "@/hooks/use-can-upload";

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
  const gate = useCanUpload();
  if (gate.kind === "ok") return <>{children}</>;
  return <PhotographerNotReadyNotice gate={gate} />;
}
