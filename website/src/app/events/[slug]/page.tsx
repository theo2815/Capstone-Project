import { Suspense, cache } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SiteHeader } from "@/components/layout/site-header";
import { fetchEventDetail } from "@/lib/api-events";
import { fetchEventPhotos } from "@/lib/api-photos";
import { EventCockpit } from "./event-cockpit";
import { PAGE_SIZE } from "@/lib/pagination-config";

interface EventPageProps {
  params: Promise<{ slug: string }>;
}

// Events change daily — dynamic SSR is the canonical render. `generateStaticParams`
// is intentionally omitted; rebuilding the site for every new event is brittle
// once /events crosses the demo dataset.
export const dynamic = "force-dynamic";

// generateMetadata and the page body run as separate calls and each needs the
// detail — without this, every SSR render hit GET /events/{slug} twice.
// cache() dedupes within one request. Kept local to this server file:
// api-events.ts is shared client code.
const getEventDetail = cache(fetchEventDetail);

export async function generateMetadata({
  params,
}: EventPageProps): Promise<Metadata> {
  const { slug } = await params;
  const event = await getEventDetail(slug);
  if (!event) return { title: "Event not found" };
  const description = `Find your photos from ${event.name}. Search by bib, browse the wall.`;
  return {
    title: event.name,
    description,
    openGraph: event.bannerUrl
      ? { title: event.name, description, images: [event.bannerUrl] }
      : undefined,
  };
}

export default async function EventPage({ params }: EventPageProps) {
  const { slug } = await params;
  const event = await getEventDetail(slug);
  if (!event) notFound();

  // Every existing event (any state, including pre-race-day upcoming) renders
  // the full cockpit. With zero photos the cockpit shows its "Photos aren't
  // ready yet" empty state + the get-notified opt-in, so an upcoming event is
  // just a no-photos cockpit — no separate pre-race-day page.
  //
  // Initial photo seed for first paint = page 0 (one Load-more page). The whole
  // envelope is threaded through so the grid header knows the true server total,
  // not the seed length. Browse mode pages via React Query from here; a bib
  // filter (Q-011 server-side) caches and pages independently.
  const initialPhotos = await fetchEventPhotos(slug, {
    offset: 0,
    limit: PAGE_SIZE.PHOTO_INCREMENT,
  });

  return (
    <main className="bg-bone text-ink min-h-screen">
      <SiteHeader />
      <Suspense fallback={<CockpitFallback />}>
        <EventCockpit event={event} initialPhotos={initialPhotos} />
      </Suspense>
    </main>
  );
}

function CockpitFallback() {
  return (
    <section
      aria-hidden="true"
      className="bg-bone min-h-[78vh] px-6 md:px-10 py-16 md:py-24 flex items-center justify-center"
    >
      <div className="w-full max-w-md">
        <div className="rounded-2xl bg-bone border border-line shadow-[0_24px_60px_-20px_rgba(17,17,17,0.18)] p-8 md:p-10 opacity-60">
          <div className="h-3 w-32 rounded-sm bg-bone-deep" />
          <div className="mt-6 h-12 w-3/4 rounded-sm bg-bone-deep" />
          <div className="mt-10 h-px w-full bg-line" />
          <div className="mt-6 h-9 w-40 rounded-full bg-bone-deep" />
        </div>
      </div>
    </section>
  );
}
