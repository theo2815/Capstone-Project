"use client";

import Link from "next/link";
import type { ListEvent } from "@/app/events/events-browser";
import { StatusChip } from "@/components/events/event-tile";
import {
  Dropdown,
  DropdownItem,
  DropdownTrigger,
} from "@/components/ui/dropdown";
import { useAdminEventOverridesStore } from "@/store/admin-event-overrides-store";
import { useToast } from "@/hooks/use-toast";

// Admin-side variant of <EventTile>. Reuses the dark banner + StatusChip
// from the runner-facing tile but trades the wrapping <Link> for an
// inline <Dropdown> of state-change actions. Admin needs the dropdown to
// be the primary affordance, not navigation.
//
// Open public gallery anchor sits as a quiet text-link inside the card
// footer so admin can still cross-check the runner-facing render.

const STATE_OPTIONS: ReadonlyArray<{
  value: ListEvent["state"];
  label: string;
}> = [
  { value: "live", label: "Live" },
  { value: "upcoming", label: "Upcoming" },
  { value: "open", label: "Recent" },
  { value: "past", label: "Archive" },
];

export function AdminEventCard({
  event,
  index = 0,
}: {
  event: ListEvent;
  index?: number;
}) {
  const setEventState = useAdminEventOverridesStore((s) => s.setEventState);
  const { showToast } = useToast();

  const dateLabel = formatShortDate(event.date);
  const cityUpper = event.city.toUpperCase();

  function handleSetState(next: ListEvent["state"]) {
    if (next === event.state) return;
    setEventState(event.id, next);
    showToast({
      kind: "success",
      message: `${event.name} → ${labelFor(next)}`,
    });
  }

  return (
    <article
      className="rounded-2xl border border-line bg-bone overflow-hidden"
      style={{
        animation: `fade-up 0.5s ${0.05 * index + 0.05}s both`,
        opacity: 0,
      }}
    >
      <div className="relative aspect-[4/3] bg-ink overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center px-6">
          <span className="font-display text-bone/25 text-2xl md:text-3xl font-medium tracking-tight text-center leading-tight">
            {event.name}
          </span>
        </div>
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
          <span className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate">
            <span className="tnum">{event.photoCount.toLocaleString()}</span>{" "}
            photos
            <span className="text-slate-soft"> · </span>
            <span className="tnum">
              {event.participantCount.toLocaleString()}
            </span>{" "}
            runners
          </span>

          <Dropdown
            align="right"
            ariaLabel={`Actions for ${event.name}`}
            trigger={
              <DropdownTrigger className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors">
                Actions ▾
              </DropdownTrigger>
            }
          >
            <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft px-3 pt-2 pb-1">
              Move to
            </p>
            {STATE_OPTIONS.map((opt) => {
              const isCurrent = opt.value === event.state;
              return (
                <DropdownItem
                  key={opt.value}
                  onClick={() => handleSetState(opt.value)}
                  disabled={isCurrent}
                >
                  {opt.label}
                  {isCurrent && (
                    <span className="ml-auto font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
                      Current
                    </span>
                  )}
                </DropdownItem>
              );
            })}
            <div className="border-t border-line my-1" />
            <Link
              href={`/events/${event.slug}`}
              target="_blank"
              rel="noreferrer"
              role="menuitem"
              className="flex w-full items-center justify-between gap-3 px-4 py-2.5 transition-colors hover:bg-bone-deep focus:bg-bone-deep focus:outline-none font-sans text-sm text-ink"
            >
              <span className="truncate text-left">
                View public gallery ↗
              </span>
            </Link>
          </Dropdown>
        </div>
      </div>
    </article>
  );
}

function labelFor(state: ListEvent["state"]): string {
  return STATE_OPTIONS.find((o) => o.value === state)?.label ?? state;
}

function formatShortDate(iso: string) {
  const d = new Date(iso + "T00:00:00");
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}
