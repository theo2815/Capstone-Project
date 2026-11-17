import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { OnboardingForm } from "@/components/onboarding/onboarding-form";

export const metadata: Metadata = {
  title: "Welcome",
};

export default function OnboardingPage() {
  return (
    <AuthShell
      rightLink={null}
      showFooter={false}
      showHorizon={false}
      wide
    >
      <OnboardingForm />
    </AuthShell>
  );
}
