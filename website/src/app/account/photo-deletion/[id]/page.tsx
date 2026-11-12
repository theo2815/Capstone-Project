"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { RequestStatus, RunnerOnly } from "@/components/account/photo-deletion/request-status";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { ProfileShellFooter } from "@/components/profile-shell";
import { ROUTES } from "@/lib/constants";

// RD-12: one deletion request's status, counts and timeline, and Withdraw.
export default function PhotoDeletionRequestPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <ProtectedRoute>
      <main className="bg-bone text-ink min-h-screen flex flex-col">
        <SiteHeader />
        <div className="flex-1 max-w-2xl mx-auto w-full px-6 md:px-10 py-10 md:py-16 flex flex-col gap-8">
          <Link href={ROUTES.ACCOUNT_PHOTO_DELETION} className="w-fit font-mono text-sm text-slate hover:text-ink">← Your deletion requests</Link>
          <RunnerOnly><RequestStatus requestId={id} /></RunnerOnly>
        </div>
        <ProfileShellFooter />
      </main>
    </ProtectedRoute>
  );
}
