import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { VerifyEmailForm } from "@/components/auth/verify-email-form";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";

// `(auth)` is a route group, so this resolves to /verify-email — the exact
// path backend EmailService builds into the registration mail. Renaming the
// folder breaks every link already sitting in someone's inbox.
export const metadata: Metadata = {
  title: "Confirm your email | QuickPitik",
};

export default function VerifyEmailPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }}>
      <Suspense fallback={<VerifyEmailSkeleton />}>
        <VerifyEmailForm />
      </Suspense>
    </AuthShell>
  );
}

function VerifyEmailSkeleton() {
  return (
    <div className="space-y-7" aria-hidden="true">
      <Kicker as="p" size="md">
        Email confirmation
      </Kicker>
      <h1 className="font-display text-5xl md:text-6xl font-medium tracking-tight leading-[1.0]">
        Confirming your
        <br />
        <span className="text-fresh">address&hellip;</span>
      </h1>
      <div className="h-12 border-b border-line" />
    </div>
  );
}
