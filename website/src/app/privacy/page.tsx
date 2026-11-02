import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { LegalDocument } from "@/components/legal/legal-document";
import { ROUTES } from "@/lib/constants";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { PRIVACY_POLICY } from "@/lib/legal/privacy";
import { assertPublishable } from "@/lib/legal/publishable";
import { LEGAL_LAST_UPDATED, LEGAL_VERSION } from "@/lib/legal/version";

// Fails a Vercel production build while draft text remains.
assertPublishable(
  [JSON.stringify(PRIVACY_POLICY), JSON.stringify(LEGAL_COPY), LEGAL_LAST_UPDATED],
  LEGAL_VERSION,
);

export const metadata: Metadata = {
  title: "Privacy Policy",
};

export default function PrivacyPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }} wide showHorizon={false}>
      <LegalDocument doc={PRIVACY_POLICY} />
    </AuthShell>
  );
}
