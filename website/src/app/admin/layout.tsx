import type { Metadata } from "next";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { AdminShell } from "@/components/admin/admin-shell";
import { AdminNotificationsBridge } from "@/components/admin/admin-notifications-bridge";

// Each admin folder's layout sets its title; templates do not chain, so this one spells out the site suffix.
export const metadata: Metadata = {
  title: { default: "Admin", template: "%s · Admin | QuickPitik" },
};

// Persistent admin layout. Mirrors /dashboard/layout.tsx — gates the entire
// /admin/* tree on ADMIN role and mounts the persistent <AdminShell>.
//
// All /admin/* routes inherit the rail + mobile chip strip.
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
