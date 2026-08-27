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
