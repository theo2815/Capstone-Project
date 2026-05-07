import { ProtectedRoute } from "@/components/auth/protected-route";
import { AdminShell } from "@/components/admin/admin-shell";

// Persistent admin layout. Mirrors /dashboard/layout.tsx — gates the entire
// /admin/* tree on ADMIN role and mounts the persistent <AdminShell>.
//
// All /admin/* routes inherit the rail + mobile chip strip + DesktopNudge.
// No FOCUSED_PATTERNS opt-out (yet) — Phase 1 surfaces all keep the shell.
// When Phase 2's /admin/disputes/[id] focused detail page lands, add an
// allowlist here the same way /dashboard/layout.tsx does for share pages.
export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute allowedRoles={["ADMIN"]}>
      <AdminShell>{children}</AdminShell>
    </ProtectedRoute>
  );
}
