import type { Metadata } from "next";
import { Suspense } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { fetchEventsList } from "@/lib/api-events";
import { EventsBrowser } from "./events-browser";

export const metadata: Metadata = {
  title: "Events | QuickPitik",
  description:
    "Browse marathon and running events in Cebu. Pick your race, find your photos.",
};

// Events change daily — opt out of static generation. The list is fetched on
// every request so admin changes (overrides + new events) propagate without
// a redeploy. See [[backend-integration/events#open-question-tile-clickability]].
export const dynamic = "force-dynamic";

export default async function EventsPage() {
  const events = await fetchEventsList();

  return (
    <main className="bg-bone text-ink relative">
      <SiteHeader />

      <Suspense fallback={null}>
        <EventsBrowser events={events} />
      </Suspense>

      <Footer />
    </main>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[12px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}
