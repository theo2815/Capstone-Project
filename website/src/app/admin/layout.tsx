"use client";

import { Sidebar, type SidebarLink } from "@/components/layout/sidebar";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { LayoutDashboard, Calendar, Users } from "lucide-react";
import { ROUTES } from "@/lib/constants";

const adminLinks: SidebarLink[] = [
  { href: ROUTES.ADMIN, label: "Overview", icon: LayoutDashboard },
  { href: ROUTES.ADMIN_EVENTS, label: "Events", icon: Calendar },
  { href: ROUTES.ADMIN_USERS, label: "Users", icon: Users },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ProtectedRoute allowedRoles={["ADMIN"]}>
      <div className="flex min-h-[calc(100vh-4rem)]">
        <Sidebar title="Admin" links={adminLinks} />
        <main className="flex-1 p-6 lg:p-8">{children}</main>
      </div>
    </ProtectedRoute>
  );
}
