"use client";

import Link from "next/link";
import type { ListEvent } from "@/app/events/events-browser";
import { StatusChip } from "@/components/events/event-tile";

// Admin-side variant of <EventTile>. Reuses the dark banner + StatusChip
// from the runner-facing tile but trades navigation for a row of explicit
// admin affordances: Edit · Delete · View ↗.
//
// Lifecycle state is now derived from `event.date` (see
// `lib/event-catalog.ts`), so the previous "Move to" dropdown is gone.
// Card body shows only Title + Date + City + Location — same fields the
// runner-facing tile shows.

interface AdminEventCardProps {
  event: ListEvent;
  index?: number;
  onEdit: (event: ListEvent) => void;
  onDelete: (event: ListEvent) => void;
}

export function AdminEventCard({
  event,
  index = 0,
  onEdit,
  onDelete,
}: AdminEventCardProps) {
  const dateLabel = formatShortDate(event.date);
  const cityUpper = event.city.toUpperCase();

  return (
    <article
      className="rounded-2xl border border-line bg-bone overflow-hidden"
      style={{
        animation: `fade-up 0.5s ${0.05 * index + 0.05}s both`,
        opacity: 0,
      }}
    >
      <div className="relative aspect-[4/3] bg-ink overflow-hidden">
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
            <span className="font-display text-bone/25 text-2xl md:text-3xl font-medium tracking-tight text-center leading-tight">
              {event.name}
            </span>
          </div>
        )}
        <StatusChip state={event.state} />
      </div>
      <div className="p-6 md:p-7">
        <p className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate mb-3">
          <span className="tnum">{dateLabel}</span> · {cityUpper}
        </p>
        <h3 className="font-display text-2xl md:text-3xl font-medium tracking-tight leading-tight text-ink">
          {event.name}
        </h3>
        <p className="mt-3 font-sans text-sm md:text-base text-ink-soft">
          {event.location}
        </p>

        <div className="mt-6 pt-4 border-t border-line flex items-center justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => onEdit(event)}
              className={actionBtn}
            >
              Edit
            </button>
            <button
              type="button"
              onClick={() => onDelete(event)}
              className={actionBtn}
              aria-label={`Delete ${event.name}`}
            >
              Delete
            </button>
          </div>
          <Link
            href={`/events/${event.slug}`}
            target="_blank"
            rel="noreferrer"
            className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-ink transition-colors"
          >
            View ↗
          </Link>
        </div>
      </div>
    </article>
  );
}

const actionBtn =
  "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors";

function formatShortDate(iso: string) {
  const d = new Date(iso + "T00:00:00");
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}
