"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { ROUTES } from "@/lib/constants";

export function OnboardingGate() {
  const router = useRouter();
  const pathname = usePathname();
  const pendingOAuth = useAuthStore((s) => s.pendingOAuth);

  useEffect(() => {
    if (pendingOAuth && pathname !== ROUTES.ONBOARDING) {
      router.replace(ROUTES.ONBOARDING);
    }
  }, [pendingOAuth, pathname, router]);

  return null;
}
