import { Suspense, cache } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SiteHeader } from "@/components/layout/site-header";
import { fetchEventDetail } from "@/lib/api-events";
import { fetchEventPhotos } from "@/lib/api-photos";
import { deriveEventState } from "@/lib/event-catalog";
import { EventCockpit } from "./event-cockpit";
import { PhotoAlertToggle } from "@/components/events/photo-alert-toggle";
import type { EventDetail } from "@/types/event";
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
  if (!event) return { title: "Event not found | QuickPitik" };
  return {
    title: `${event.name} | QuickPitik`,
    description: `Find your photos from ${event.name}. Search by bib, browse the wall.`,
  };
}

export default async function EventPage({ params }: EventPageProps) {
  const { slug } = await params;
  const event = await getEventDetail(slug);
  if (!event) notFound();

  // Pre-race-day events stay viewable as a public page but the gallery is
  // gated behind race day. Show a notice with the cover + metadata and a
  // clear "Opens on [date]" call-out so runners (and admin previewing
  // their creation) see something instead of a 404.
  if (deriveEventState(event.date) === "upcoming") {
    return (
      <main className="bg-bone text-ink min-h-screen">
        <SiteHeader />
        <UpcomingEventNotice event={event} />
      </main>
    );
  }

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

function UpcomingEventNotice({ event }: { event: EventDetail }) {
  const raceDay = new Date(`${event.date}T00:00:00`);
  const dateLabel = raceDay.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
  const cityUpper = (event.location.split(",").pop() ?? event.location)
    .trim()
    .toUpperCase();
  return (
    <section className="px-6 md:px-10 py-16 md:py-24">
      <div className="mx-auto max-w-3xl">
        <div className="relative aspect-[16/9] rounded-2xl overflow-hidden bg-ink">
          {event.bannerUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={event.bannerUrl}
              alt=""
              aria-hidden="true"
              className="absolute inset-0 w-full h-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center px-6">
              <span className="font-display text-bone/25 text-3xl md:text-5xl font-medium tracking-tight text-center leading-tight">
                {event.name}
              </span>
            </div>
          )}
        </div>
        <div className="mt-10">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-fresh">
            Opens · <span className="tnum">{dateLabel}</span>
          </p>
          <h1 className="mt-4 font-display font-extrabold text-3xl md:text-5xl tracking-tight leading-[1.05] text-ink">
            {event.name}
          </h1>
          <p className="mt-4 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            {cityUpper}
          </p>
          <p className="mt-2 font-sans text-base text-ink-soft max-w-prose">
            {event.location}
          </p>
          <p className="mt-10 font-sans text-base md:text-lg text-ink-soft max-w-prose">
            The gallery and runner search open on race day. Photographers have
            a four-day window from race day to upload — check back then to
            find your photos.
          </p>
          <div className="mt-2 max-w-md">
            <PhotoAlertToggle eventSlug={event.slug} />
          </div>
        </div>
      </div>
    </section>
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
