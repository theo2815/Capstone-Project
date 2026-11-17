"use client";

import { Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { RequestForm } from "@/components/account/photo-deletion/request-form";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { ProfileShellFooter } from "@/components/profile-shell";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES } from "@/lib/constants";

// RD-11: a runner asks to delete their photos from one event. No allowedRoles:
// the form itself tells other roles to email us instead of redirecting them.
export default function NewPhotoDeletionPage() {
  return (
    <ProtectedRoute>
      <main className="bg-bone text-ink min-h-screen flex flex-col">
        <SiteHeader />
        <div className="flex-1 max-w-2xl mx-auto w-full px-6 md:px-10 py-10 md:py-16 flex flex-col gap-8">
          <Link href={ROUTES.ACCOUNT} className="w-fit font-mono text-sm text-slate hover:text-ink">← Account</Link>
          <header className="flex flex-col gap-3">
            <Kicker as="p">Photo deletion</Kicker>
            <h1 className="font-display text-3xl md:text-4xl font-extrabold leading-tight">Remove your photos from an event.</h1>
            <p className="max-w-xl font-sans text-base text-slate leading-relaxed">
              A short live check finds your photos in one event. Very close matches can be hidden right away; our team reviews
              every photo before anything is deleted.
            </p>
          </header>
          {/* useSearchParams needs a Suspense boundary for the static prerender. */}
          <Suspense fallback={null}>
            <FormWithEvent />
          </Suspense>
        </div>
        <ProfileShellFooter />
      </main>
    </ProtectedRoute>
  );
}

function FormWithEvent() {
  const slug = useSearchParams().get("event");
  return <RequestForm eventSlug={slug || null} />;
}
