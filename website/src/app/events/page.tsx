import type { Metadata } from "next";
import { Suspense } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { fetchPublicEventsCatalog } from "@/lib/api-events";
import { EventsBrowser } from "./events-browser";
import { Footer } from "@/components/layout/footer";

export const metadata: Metadata = {
  title: "Events",
  description:
    "Browse marathon and running events in Cebu. Pick your race, find your photos.",
};

// Events change daily — opt out of static generation. The list is fetched on
// every request so admin changes (overrides + new events) propagate without
// a redeploy. See [[backend-integration/events#open-question-tile-clickability]].
export const dynamic = "force-dynamic";

export default async function EventsPage() {
  const events = await fetchPublicEventsCatalog().catch(() => undefined);

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

