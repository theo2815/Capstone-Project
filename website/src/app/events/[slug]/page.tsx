import { Suspense } from "react";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import { SiteHeader } from "@/components/layout/site-header";
import { MOCK_EVENT_DETAILS, getEventBySlug } from "../mock-events";
import { generateMockPhotos } from "./mock-photos";
import { EventCockpit } from "./event-cockpit";

interface EventPageProps {
  params: Promise<{ slug: string }>;
}

export async function generateMetadata({
  params,
}: EventPageProps): Promise<Metadata> {
  const { slug } = await params;
  const event = getEventBySlug(slug);
  if (!event) return { title: "Event not found | QuickPitik" };
  return {
    title: `${event.name} | QuickPitik`,
    description: `Find your photos from ${event.name}. Search by bib, browse the wall.`,
  };
}

export async function generateStaticParams() {
  return Object.keys(MOCK_EVENT_DETAILS).map((slug) => ({ slug }));
}

export default async function EventPage({ params }: EventPageProps) {
  const { slug } = await params;
  const event = getEventBySlug(slug);
  if (!event) notFound();

  const photos = generateMockPhotos(event);

  return (
    <main className="bg-bone text-ink min-h-screen">
      <SiteHeader />
      <Suspense fallback={<CockpitFallback />}>
        <EventCockpit event={event} photos={photos} />
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
