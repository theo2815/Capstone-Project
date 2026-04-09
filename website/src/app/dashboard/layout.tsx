"use client";

import { Sidebar, type SidebarLink } from "@/components/layout/sidebar";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { LayoutDashboard, Calendar, DollarSign } from "lucide-react";
import { ROUTES } from "@/lib/constants";

const dashboardLinks: SidebarLink[] = [
  { href: ROUTES.DASHBOARD, label: "Overview", icon: LayoutDashboard },
  { href: ROUTES.DASHBOARD_EVENTS, label: "My Events", icon: Calendar },
  { href: ROUTES.DASHBOARD_EARNINGS, label: "Earnings", icon: DollarSign },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute allowedRoles={["PHOTOGRAPHER"]}>
      <div className="flex min-h-[calc(100vh-4rem)]">
        <Sidebar title="Dashboard" links={dashboardLinks} />
        <main className="flex-1 p-6 lg:p-8">{children}</main>
      </div>
    </ProtectedRoute>
  );
}
