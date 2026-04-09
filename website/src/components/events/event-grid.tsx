import { EventCard } from "./event-card";
import type { Event } from "@/types/event";

interface EventGridProps {
  events: Event[];
}

export function EventGrid({ events }: EventGridProps) {
  if (events.length === 0) {
    return (
      <div className="py-12 text-center">
        <p className="text-gray-500">No events found.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
      {events.map((event) => (
        <EventCard key={event.id} event={event} />
      ))}
    </div>
  );
}
