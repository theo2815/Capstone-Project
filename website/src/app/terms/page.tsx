import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { LegalDocument } from "@/components/legal/legal-document";
import { ROUTES } from "@/lib/constants";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { assertPublishable } from "@/lib/legal/publishable";
import { TERMS_OF_SERVICE } from "@/lib/legal/terms";
import { LEGAL_LAST_UPDATED, LEGAL_VERSION } from "@/lib/legal/version";

// Fails a Vercel production build while draft text remains.
assertPublishable(
  [JSON.stringify(TERMS_OF_SERVICE), JSON.stringify(LEGAL_COPY), LEGAL_LAST_UPDATED],
  LEGAL_VERSION,
);

export const metadata: Metadata = {
  title: "Terms of Service",
};

export default function TermsPage() {
  return (
    <AuthShell rightLink={{ label: "Sign in", href: ROUTES.LOGIN }} wide showHorizon={false}>
      <LegalDocument doc={TERMS_OF_SERVICE} />
    </AuthShell>
  );
}
