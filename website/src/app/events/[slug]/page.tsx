import { Suspense } from "react";
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

export async function generateMetadata({
  params,
}: EventPageProps): Promise<Metadata> {
  const { slug } = await params;
  const event = await fetchEventDetail(slug);
  if (!event) return { title: "Event not found | QuickPitik" };
  return {
    title: `${event.name} | QuickPitik`,
    description: `Find your photos from ${event.name}. Search by bib, browse the wall.`,
  };
}

export default async function EventPage({ params }: EventPageProps) {
  const { slug } = await params;
  const event = await fetchEventDetail(slug);
  if (!event) notFound();

  // Initial photo seed for first paint. Browse mode refetches via React Query
  // when the user enters a bib (Q-011 server-side filter).
  const initialPhotos = await fetchEventPhotos(slug, {
    offset: 0,
    limit: PAGE_SIZE.PHOTO_INITIAL * 2,
  });

  return (
    <main className="bg-bone text-ink min-h-screen">
      <SiteHeader />
      <Suspense fallback={<CockpitFallback />}>
        <EventCockpit event={event} initialPhotos={initialPhotos.items} />
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
