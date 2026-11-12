"use client";

import Link from "next/link";
import { countsLine, RunnerOnly } from "@/components/account/photo-deletion/request-status";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { ProfileShellFooter } from "@/components/profile-shell";
import { BTN_GHOST, BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { useRunnerRemovalRequests } from "@/hooks/use-photo-removal";
import { ROUTES } from "@/lib/constants";
import { formatLongDate } from "@/lib/format";
import { runnerStatusLabel } from "@/lib/runner-removal-status";
import { cn } from "@/lib/utils";

// RD-12: the runner's deletion requests, newest first.
export default function PhotoDeletionRequestsPage() {
  return (
    <ProtectedRoute>
      <main className="bg-bone text-ink min-h-screen flex flex-col">
        <SiteHeader />
        <div className="flex-1 max-w-2xl mx-auto w-full px-6 md:px-10 py-10 md:py-16 flex flex-col gap-8">
          <Link href={ROUTES.ACCOUNT} className="w-fit font-mono text-sm text-slate hover:text-ink">← Account</Link>
          <header className="flex flex-col gap-3">
            <Kicker as="p">Photo deletion</Kicker>
            <h1 className="font-display text-3xl md:text-4xl font-extrabold leading-tight">Your deletion requests.</h1>
          </header>
          <RunnerOnly><RequestList /></RunnerOnly>
        </div>
        <ProfileShellFooter />
      </main>
    </ProtectedRoute>
  );
}

function RequestList() {
  // shortcut: first page only (the server's default page size); a runner holds at most 3 open requests. Add paging if closed ones pile up.
  const query = useRunnerRemovalRequests();
  const newRequest = (
    <Link href={ROUTES.ACCOUNT_PHOTO_DELETION_NEW} className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-fit")}>Request photo deletion →</Link>
  );
  if (query.isPending) return <p role="status" className="font-sans text-sm text-slate">Loading your requests…</p>;
  if (!query.data) {
    return (
      <div role="alert" className="flex flex-col items-start gap-3 font-sans text-sm text-error">
        <p>We couldn&apos;t load your requests.</p>
        <button type="button" onClick={() => void query.refetch()} className={cn(BTN_GHOST, BTN_SIZE.sm)}>Retry</button>
      </div>
    );
  }
  if (query.data.items.length === 0) {
    return (
      <div className="flex flex-col gap-5">
        <p className="font-sans text-base text-slate">You haven&apos;t asked us to delete any photos yet.</p>
        {newRequest}
      </div>
    );
  }
  return (
    <div className="flex flex-col gap-8">
      <ul className="divide-y divide-line border-y border-line">
        {query.data.items.map((r) => (
          <li key={r.id}>
            <Link href={`${ROUTES.ACCOUNT_PHOTO_DELETION}/${encodeURIComponent(r.id)}`}
              className="flex flex-col gap-1.5 px-2 py-4 hover:bg-bone-deep/50 transition-colors">
              <span className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
                <span className="font-display text-lg font-bold text-ink">{r.eventName}</span>
                <Kicker as="span" tone="soft" tnum>{formatLongDate(r.createdAt)}</Kicker>
              </span>
              <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink">
                {runnerStatusLabel(r.status)}
              </span>
              <span className="font-mono tnum text-sm text-slate">{countsLine(r.counts)}</span>
            </Link>
          </li>
        ))}
      </ul>
      {newRequest}
    </div>
  );
}
