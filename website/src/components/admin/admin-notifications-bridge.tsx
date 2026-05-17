"use client";

import { useAuth } from "@/hooks/use-auth";
import { useAdminNotificationsWs } from "@/hooks/use-admin-notifications-ws";

// Invisible mount point for the admin WebSocket. Sits inside AdminLayout's
// <ProtectedRoute allowedRoles={["ADMIN"]}> guard, so the hook only runs
// after the role check has gated this branch — but we still pass `enabled`
// derived from auth state so the hook self-exits on a signed-out moment
// between renders (refresh-token failure, manual logout).
export function AdminNotificationsBridge() {
  const { user, isAuthenticated, isLoading } = useAuth();
  const enabled = isAuthenticated && !isLoading && user?.role === "ADMIN";
  useAdminNotificationsWs(enabled);
  return null;
}
