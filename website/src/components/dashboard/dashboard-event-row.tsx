import Link from "next/link";
import { LogoPlaceholder } from "@/components/ui/logo-placeholder";
import type { EventState, ListEvent } from "@/types/event";
import { ROUTES } from "@/lib/constants";

// A photographer's event as one compact list row (~80px), replacing the
// 4:3 cover tile on /dashboard/events so a screen holds 8–9 races instead of
// about 1.5. Same destination as the old manage tile: the event's page.
// Sentence case for details; mono only for the counts.
export function DashboardEventRow({
  event,
  photoCount,
  salesCount,
  note,
  href,
  cta = "Open",
  ctaMuted = false,
  ariaLabel,
}: {
  event: ListEvent;
  photoCount: number;
  /** Left out where sales aren't known (the upload picker's shared races). */
  salesCount?: number;
  note?: string;
  /** Defaults to the event's own page; the upload picker points at the upload form. */
  href?: string;
  cta?: string;
  ctaMuted?: boolean;
  ariaLabel?: string;
}) {
  return (
    <Link
      href={href ?? `${ROUTES.DASHBOARD_EVENTS}/${event.id}`}
      aria-label={ariaLabel}
      className="group flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 transition-colors hover:bg-bone-deep/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-fresh"
    >
      <div className="relative w-[88px] h-[56px] shrink-0 rounded-lg overflow-hidden bg-bone-deep flex items-center justify-center">
        <LogoPlaceholder size="sm" className="absolute inset-0" />
        {event.bannerUrl && (
          // Hidden on error (e.g. an expired signed URL) so the logo behind it shows.
          // eslint-disable-next-line @next/next/no-img-element
          <img src={event.bannerUrl} alt="" aria-hidden="true" onError={(e) => { e.currentTarget.hidden = true; }} className="absolute inset-0 w-full h-full object-cover" />
        )}
      </div>
      <div className="min-w-0 flex-1 basis-[220px]">
        <h3 className="font-display text-base font-semibold text-ink group-hover:text-fresh transition-colors">
          {event.name}
        </h3>
        <p className="font-sans text-sm text-slate">
          {formatDate(event.date)} · {event.location || event.city}
          {note && <> · <span className="text-ink-soft">{note}</span></>}
        </p>
      </div>
      <span className={`inline-flex items-center gap-1.5 font-sans text-sm w-[120px] shrink-0 ${STATE_TEXT[event.state]}`}>
        <span aria-hidden="true" className="size-[7px] rounded-full bg-current" />
        {STATE_LABEL[event.state]}
      </span>
      <span className="font-sans text-sm text-slate w-[150px] shrink-0">
        <span className="font-mono text-[13px] text-ink tnum">{photoCount.toLocaleString()}</span> photos
        {salesCount !== undefined && <>
          {" "}·{" "}<span className="font-mono text-[13px] text-ink tnum">{salesCount.toLocaleString()}</span> sold
        </>}
      </span>
      {/* The whole row is the link; the label only helps on wider screens, unless it says
          something the row doesn't (the upload picker's "Upload window closed"). */}
      <span className={`ml-auto font-sans text-sm items-center gap-1.5 min-h-[44px] ${ctaMuted ? "text-slate-soft inline-flex" : "text-slate group-hover:text-ink hidden sm:inline-flex"}`}>
        {cta.replace(/ →$/, "")} {!ctaMuted && <span aria-hidden="true">→</span>}
      </span>
    </Link>
  );
}

const STATE_LABEL: Record<EventState, string> = {
  live: "Live now",
  upcoming: "Upcoming",
  open: "Open for sale",
  past: "Archived",
};
// The dot takes the text colour; the word carries the meaning.
const STATE_TEXT: Record<EventState, string> = {
  live: "text-fresh",
  upcoming: "text-slate",
  open: "text-ink",
  past: "text-slate-soft",
};


function formatDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  return `${d.toLocaleString("en-US", { month: "short" })} ${d.getDate()}, ${d.getFullYear()}`;
}
