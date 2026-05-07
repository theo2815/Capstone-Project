import { redirect } from "next/navigation";
import { ROUTES } from "@/lib/constants";

// Phase 1 admin redesign — /admin lands directly on the Inbox queue. The
// previous overview content (8-tile KPI grid + 30-day trend + decisions
// timeline) lives at /admin/overview as the weekly-review tab.
//
// Server-side redirect runs before <ProtectedRoute> in the layout. The
// inbox layout (same AdminLayout) re-applies the ADMIN role gate, so
// auth still gets enforced — just at the inbox URL instead of /admin.
export default function AdminPage() {
  redirect(ROUTES.ADMIN_INBOX);
}
