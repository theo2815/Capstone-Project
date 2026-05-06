"use client";

import { usePathname } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { DashboardShell } from "@/components/dashboard/dashboard-shell";

// Photographer-only gate + persistent dashboard shell. Most /dashboard/*
// pages render inside DashboardShell (SiteHeader + sticky rail + content).
// A small list of "focused mode" routes opt OUT of the shell so they can
// render their own minimal chrome — currently the per-event share page at
// /dashboard/events/[id] (NOT /dashboard/events/[id]/photos, which keeps
// the rail). Add new patterns below sparingly; the rail is the default.
const FOCUSED_PATTERNS: ReadonlyArray<RegExp> = [
  /^\/dashboard\/events\/[^/]+\/?$/,
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname() ?? "";
  const isFocused = FOCUSED_PATTERNS.some((re) => re.test(pathname));

  return (
    <ProtectedRoute allowedRoles={["PHOTOGRAPHER"]}>
      {isFocused ? children : <DashboardShell>{children}</DashboardShell>}
    </ProtectedRoute>
  );
}
