"use client";

import type { EventState } from "@/app/events/events-browser";
import { Slab } from "@/components/profile-shell";
import { VerificationBanner } from "@/components/dashboard/verification-banner";
import { EventTile } from "@/components/events/event-tile";
import { useCanUpload } from "@/hooks/use-can-upload";
import { EVENT_CATALOG } from "@/lib/event-catalog";

// Photographer upload picker. Shows every event in the catalog (not just the
// photographer's own coverage) so they can pick where to lift photos to. The
// per-photographer view of "events I covered" lives at /dashboard/events.

const SECTIONS: Array<{
  id: string;
  number: string;
  title: string;
  caption: string;
  matchState: EventState;
}> = [
  {
    id: "live",
    number: "01",
    title: "Live now",
    caption: "Race day · upload as you shoot",
    matchState: "live",
  },
  {
    id: "upcoming",
    number: "02",
    title: "Upcoming",
    caption: "Pre-stage and test your kit",
    matchState: "upcoming",
  },
  {
    id: "recent",
    number: "03",
    title: "Recent",
    caption: "Last 30 days · late frames welcome",
    matchState: "open",
  },
  {
    id: "archive",
    number: "04",
    title: "Archive",
    caption: "Older races · keep adding if you have them",
    matchState: "past",
  },
];

export default function UploadPickerPage() {
  const gate = useCanUpload();
  const canUpload = gate.kind === "ok";

  const visibleSections = SECTIONS.map((section) => ({
    ...section,
    items: EVENT_CATALOG.filter((e) => e.state === section.matchState),
  })).filter((section) => section.items.length > 0);

  if (visibleSections.length === 0) {
    return (
      <>
        <VerificationBanner />
        <PickerEmpty />
      </>
    );
  }

  return (
    <>
      <VerificationBanner />

      {visibleSections.map((section) => {
        const noun = section.items.length === 1 ? "race" : "races";
        return (
          <Slab
            key={section.id}
            id={section.id}
            number={section.number}
            title={section.title}
            caption={section.caption}
            trailing={`${section.items.length} ${noun}`}
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 md:gap-8">
              {section.items.map((event, i) => (
                <EventTile
                  key={event.id}
                  mode="upload"
                  event={event}
                  index={i}
                  canUpload={canUpload}
                />
              ))}
            </div>
          </Slab>
        );
      })}
    </>
  );
}

function PickerEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No events to upload to.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Schedule a coverage to start lifting photos.
      </p>
    </div>
  );
}
