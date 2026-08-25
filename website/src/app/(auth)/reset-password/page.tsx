import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { ResetPasswordForm } from "@/components/auth/reset-password-form";
import { ROUTES } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Reset password | QuickPitik",
};

export default function ResetPasswordPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }}>
      <Suspense fallback={<ResetPasswordSkeleton />}>
        <ResetPasswordForm />
      </Suspense>
    </AuthShell>
  );
}

function ResetPasswordSkeleton() {
  return (
    <div className="space-y-7" aria-hidden="true">
      <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate">
        Reset access
      </p>
      <h1 className="font-hero text-5xl md:text-6xl">
        Set a new
        <br />
        <span className="text-fresh">password.</span>
      </h1>
      <div className="space-y-6 pt-2">
        <div className="h-12 border-b border-line" />
        <div className="h-12 border-b border-line" />
      </div>
    </div>
  );
}
