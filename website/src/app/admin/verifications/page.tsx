"use client";

import {
  VerificationsQueue,
  usePendingVerificationsCount,
} from "@/components/admin/verifications-queue";
import { Kicker } from "@/components/ui/kicker";

// Phase 1 admin redesign: /admin/verifications continues to exist as a
// deep-link / focused-mode route, but the rail surfaces the queue under
// /admin/inbox (which adds the KPI strip and, in Phase 2, type filter
// chips for disputes/flags/payouts). The verifications page itself just
// renders the dedicated header + the shared <VerificationsQueue>.

export default function AdminVerificationsPage() {
  const pendingCount = usePendingVerificationsCount();

  return (
    <>
      <Header pendingCount={pendingCount} />
      <VerificationsQueue />
    </>
  );
}

function Header({ pendingCount }: { pendingCount: number }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Verifications · <span className="tnum">{pendingCount}</span> waiting
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Verifications.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Review what each photographer set up and decide if they&apos;re ready to
        sell on QuickPitik. Approve closes the loop in one tap; reject
        sends them back with a reason.
      </p>
    </header>
  );
}
