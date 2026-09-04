"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, useState, type ReactNode } from "react";
import { ApiError } from "@/lib/api";
import { ToastContainer } from "@/components/ui/toast";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { FloatingCart } from "@/components/cart/floating-cart";
import { CheckoutResumeWatcher } from "@/components/cart/checkout-resume-watcher";
import { AuthHydrator } from "@/components/layout/auth-hydrator";
import { OnboardingGate } from "@/components/onboarding/onboarding-gate";

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            // Focus refetch is off: every surface that needs freshness has
            // its own channel (WS pushes, the admin adaptive poll, the
            // explicit focus syncs in verification/admin-users/inbox stores).
            // With it on, every tab return refired every mounted query —
            // on /admin/inbox that was five list fetches per alt-tab.
            refetchOnWindowFocus: false,
            // Dashboard/admin navigation is a tab loop; the 5-min default
            // discarded caches on a coffee break. Safe to hold longer since
            // every auth transition clears the whole cache (use-auth).
            gcTime: 30 * 60_000,
            // Never auto-retry a 429 — the bucket is empty and an immediate
            // retry just deepens the denial (rate limiting is ON by default
            // backend-side since 2026-08-27).
            retry: (failureCount, error) =>
              !(error instanceof ApiError && error.status === 429) &&
              failureCount < 1,
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={queryClient}>
      <AuthHydrator />
      <OnboardingGate />
      {children}
      <ToastContainer />
      <ConfirmationOverlay />
      <Suspense fallback={null}>
        <FloatingCart />
      </Suspense>
      <Suspense fallback={null}>
        <CheckoutResumeWatcher />
      </Suspense>
    </QueryClientProvider>
  );
}
