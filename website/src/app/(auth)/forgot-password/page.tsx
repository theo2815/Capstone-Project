import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { ForgotPasswordForm } from "@/components/auth/forgot-password-form";
import { ROUTES } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Forgot password | QuickPitik",
};

export default function ForgotPasswordPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }}>
      <ForgotPasswordForm />
    </AuthShell>
  );
}
