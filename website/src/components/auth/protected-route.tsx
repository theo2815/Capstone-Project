"use client";

import { useEffect, type ReactNode } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { buildLoginRedirect, roleHome } from "@/lib/redirect";
import { Spinner } from "@/components/ui/spinner";
import type { Role } from "@/types/user";

interface ProtectedRouteProps {
  children: ReactNode;
  allowedRoles?: Role[];
}

export function ProtectedRoute({
  children,
  allowedRoles,
}: ProtectedRouteProps) {
  const router = useRouter();
  const { user, isAuthenticated, isLoading } = useAuth();

  useEffect(() => {
    if (isLoading) return;

    if (!isAuthenticated) {
      const currentUrl =
        typeof window !== "undefined"
          ? window.location.pathname + window.location.search
          : "";
      router.replace(buildLoginRedirect(currentUrl));
      return;
    }

    if (allowedRoles && user && !allowedRoles.includes(user.role)) {
      router.replace(roleHome(user.role));
    }
  }, [isLoading, isAuthenticated, user, allowedRoles, router]);

  if (isLoading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }

  if (!isAuthenticated) return null;
  if (allowedRoles && user && !allowedRoles.includes(user.role)) return null;

  return <>{children}</>;
}
