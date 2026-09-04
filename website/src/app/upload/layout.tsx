"use client";

import { ProtectedRoute } from "@/components/auth/protected-route";

// Photographer-only gate for /upload/[eventId]. Its sibling picker
// (/dashboard/upload) is gated by dashboard/layout.tsx, but this route had no
// guard at all — a guest loaded the full photographer-shaped page and sat on
// "Uploads disabled" forever instead of being bounced to /login. No data
// leaked (the event catalog is public; useCanUpload reads empty stores), so
// this closes a gating inconsistency, not an access hole.
export default function UploadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute allowedRoles={["PHOTOGRAPHER"]}>{children}</ProtectedRoute>
  );
}
