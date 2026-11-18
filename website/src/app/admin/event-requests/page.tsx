"use client";

import { EventRequestsQueue } from "@/components/admin/event-requests-queue";
import { Kicker } from "@/components/ui/kicker";
import { Skeleton } from "@/components/ui/skeleton";
import { useAdminEvents } from "@/hooks/use-admin-events";

// Focus-mode route for the V46 event requests queue, sharing its body with
// /admin/inbox?type=events via the lifted <EventRequestsQueue>. Same shape
// as /admin/disputes.

export default function AdminEventRequestsPage() {
  // ER1: count the list below (the same cached query), not the KPI figure, so the two never disagree.
  const query = useAdminEvents({ review: "queue" });

  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        {/* A div, not a p: the loading Skeleton is a div. */}
        <Kicker as="div">
          Event requests
          {query.data ? (
            <> · <span className="tnum">{query.data.length}</span> waiting</>
          ) : query.isPending ? (
            <div aria-busy="true" className="inline-block align-middle ml-2">
              <Skeleton className="h-3 w-20" />
            </div>
          ) : null}
        </Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Event requests.
        </h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Photographer events stay hidden, and pricing changes on live events
          stay pending, until you approve them.
        </p>
      </header>
      <EventRequestsQueue />
    </>
  );
}
