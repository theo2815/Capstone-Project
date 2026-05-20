"use client";

import { useAuth } from "@/hooks/use-auth";
import { useAdminNotificationsWs } from "@/hooks/use-admin-notifications-ws";
import { useAdminQueueRealtime } from "@/hooks/use-admin-queue-realtime";

// Invisible mount point for the admin WebSocket + adaptive polling
// fallback. Sits inside AdminLayout's <ProtectedRoute allowedRoles=
// {["ADMIN"]}> guard, so the hooks only run after the role check has
// gated this branch — but we still pass `enabled` derived from auth
// state so they self-exit on a signed-out moment between renders
// (refresh-token failure, manual logout).
//
// useAdminNotificationsWs is the primary path (push). useAdminQueueRealtime
// is the safety net (poll): 5 min when WS is healthy, 30 s when degraded,
// so an admin queue never silently goes stale if the WS drops. A-2.
export function AdminNotificationsBridge() {
  const { user, isAuthenticated, isLoading } = useAuth();
  const enabled = isAuthenticated && !isLoading && user?.role === "ADMIN";
  useAdminNotificationsWs(enabled);
  useAdminQueueRealtime(enabled);
  return null;
}
