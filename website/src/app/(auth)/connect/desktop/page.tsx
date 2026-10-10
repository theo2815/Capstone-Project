import { Suspense } from "react";
import type { Metadata } from "next";
import { AuthShell } from "@/components/auth/auth-shell";
import { ConnectDesktop } from "@/components/auth/connect-desktop";
import { ROUTES } from "@/lib/constants";

export const metadata: Metadata = {
  title: "Connect BatchMyPhotos | QuickPitik",
  robots: { index: false, follow: false },
};

export default function ConnectDesktopPage() {
  return (
    <AuthShell rightLink={{ label: "Dashboard", href: ROUTES.DASHBOARD }} showFooter={false}>
      <Suspense fallback={null}>
        <ConnectDesktop />
      </Suspense>
    </AuthShell>
  );
}
