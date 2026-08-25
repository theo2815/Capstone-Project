import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { LoginForm } from "@/components/auth/login-form";
import { ROUTES } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Log in | QuickPitik",
};

export default function LoginPage() {
  return (
    <AuthShell rightLink={{ label: "Sign up", href: ROUTES.REGISTER }}>
      <Suspense fallback={<LoginSkeleton />}>
        <LoginForm />
      </Suspense>
    </AuthShell>
  );
}

function LoginSkeleton() {
  return (
    <div className="space-y-7" aria-hidden="true">
      <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate">
        Log in
      </p>
      <h1 className="font-hero text-5xl md:text-6xl">
        Welcome
        <br />
        <span className="text-fresh">back.</span>
      </h1>
      <div className="space-y-6 pt-2">
        <div className="h-12 border-b border-line" />
        <div className="h-12 border-b border-line" />
      </div>
    </div>
  );
}
