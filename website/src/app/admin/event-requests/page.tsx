"use client";

import { EventRequestsQueue } from "@/components/admin/event-requests-queue";
import { Kicker } from "@/components/ui/kicker";
import { useAdminKpis } from "@/hooks/use-admin-data";

// Focus-mode route for the V46 event requests queue, sharing its body with
// /admin/inbox?type=events via the lifted <EventRequestsQueue>. Same shape
// as /admin/disputes.

export default function AdminEventRequestsPage() {
  const waiting = useAdminKpis()?.pendingEventRequests ?? 0;

  return (
    <>
      <header className="pb-8 md:pb-12 border-b border-line">
        <Kicker as="p">
          Event requests · <span className="tnum">{waiting}</span> waiting
        </Kicker>
        <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
          Event requests.
        </h1>
        <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
          Photographer-created events stay invisible until you approve them.
          Pricing changes on a live event wait here too — the gallery keeps
          its current settings until you apply the change.
        </p>
      </header>
      <EventRequestsQueue />
    </>
  );
}
