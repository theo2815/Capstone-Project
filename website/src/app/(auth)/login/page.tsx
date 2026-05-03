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
      <LoginForm />
    </AuthShell>
  );
}
