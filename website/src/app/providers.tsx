"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Suspense, useState, type ReactNode } from "react";
import { ToastContainer } from "@/components/ui/toast";
import { ConfirmationOverlay } from "@/components/ui/confirmation-overlay";
import { FloatingCart } from "@/components/cart/floating-cart";
import { AuthHydrator } from "@/components/layout/auth-hydrator";
import { OnboardingGate } from "@/components/onboarding/onboarding-gate";

export function Providers({ children }: { children: ReactNode }) {
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60 * 1000,
            retry: 1,
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
    </QueryClientProvider>
  );
}
