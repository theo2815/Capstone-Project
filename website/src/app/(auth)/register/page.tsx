import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { RegisterForm } from "@/components/auth/register-form";
import { ROUTES } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Create account | QuickPitik",
};

export default function RegisterPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }}>
      <Suspense fallback={<RegisterSkeleton />}>
        <RegisterForm />
      </Suspense>
    </AuthShell>
  );
}

function RegisterSkeleton() {
  return (
    <div className="space-y-7" aria-hidden="true">
      <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
        Create account
      </p>
      <h1 className="font-hero text-5xl md:text-6xl">
        Join
        <br />
        <span className="text-fresh">QuickPitik.</span>
      </h1>
      <div className="space-y-6 pt-2">
        <div className="h-12 border-b border-line" />
        <div className="h-12 border-b border-line" />
        <div className="h-12 border-b border-line" />
      </div>
    </div>
  );
}
