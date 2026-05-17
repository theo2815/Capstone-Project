import { ProtectedRoute } from "@/components/auth/protected-route";
import { AdminShell } from "@/components/admin/admin-shell";
import { AdminNotificationsBridge } from "@/components/admin/admin-notifications-bridge";

// Persistent admin layout. Mirrors /dashboard/layout.tsx — gates the entire
// /admin/* tree on ADMIN role and mounts the persistent <AdminShell>.
//
// All /admin/* routes inherit the rail + mobile chip strip + DesktopNudge.
// No FOCUSED_PATTERNS opt-out (yet) — Phase 1 surfaces all keep the shell.
// When Phase 2's /admin/disputes/[id] focused detail page lands, add an
// allowlist here the same way /dashboard/layout.tsx does for share pages.
//
// AdminNotificationsBridge mounts the admin WS hook (no UI) so push events
// — verification_submitted, dispute_filed, payout_report_filed — invalidate
// the relevant admin stores in real time without a manual refresh.
export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute allowedRoles={["ADMIN"]}>
      <AdminNotificationsBridge />
      <AdminShell>{children}</AdminShell>
    </ProtectedRoute>
  );
}
