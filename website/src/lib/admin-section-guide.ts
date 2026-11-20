import type { SectionGuideEntry } from "@/lib/section-guide";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// Mirror of `SECTION_GUIDE` for the photographer dashboard, but for the
// admin workspace. Single source of truth for "what each rail tab does"
// — consumed by `<AdminRail>` per-row tooltips.
//
// Entries follow the rail's order and numbers (admin-rail.tsx): Inbox,
// then the queues in ADMIN_QUEUES order, then Catalog, Reports and Account.

const ADMIN_SECTION_GUIDE_ALL: ReadonlyArray<SectionGuideEntry> = [
  {
    number: "01",
    href: ROUTES.ADMIN_INBOX,
    label: "Inbox",
    body: "Your morning landing: every queue waiting on an admin, one chip at a time.",
  },
  {
    number: "02",
    href: ROUTES.ADMIN_DISPUTES,
    label: "Disputes",
    body: "Refund requests and complaints from runners. Resolve before they escalate.",
  },
  {
    number: "03",
    href: ROUTES.ADMIN_FLAGS,
    label: "Flags",
    body: "Photos reported by runners or surfaced during review. Hide what shouldn't be live.",
  },
  {
    number: "04",
    href: ROUTES.ADMIN_VERIFICATIONS,
    label: "Verifications",
    body: "New photographers waiting for approval, and the seller details they sent for an ID check.",
  },
  {
    number: "05",
    href: ROUTES.ADMIN_EVENT_REQUESTS,
    label: "Event requests",
    body: "Photographer-created events waiting on a decision, and pricing changes parked on live ones. Approve, or send back with a reason.",
  },
  {
    number: "06",
    href: ROUTES.ADMIN_PAYOUTS,
    label: "Payouts",
    body: "Payout requests across all photographers. Mark one paid or hold it.",
  },
  {
    number: "07",
    href: ROUTES.ADMIN_RELEVANCE,
    label: "Photo Relevance",
    body: "Batches checked after upload for junk, wrong-event and copied photos. Read each verdict, its reason and its sample photos.",
  },
  {
    number: "08",
    href: ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS,
    label: "Photo removals",
    body: "Runner requests and support requests. Review what was hidden or suggested, find their photos across every photographer in one event and verify which to hide.",
  },
  {
    number: "09",
    href: ROUTES.ADMIN_EVENTS,
    label: "Events",
    body: "Every race on the platform. Edit lifecycle state, archive, or unhide.",
  },
  {
    number: "10",
    href: ROUTES.ADMIN_PHOTOGRAPHERS,
    label: "Photographers",
    body: "Directory of approved + pending photographers. Drill in for history and brand checks.",
  },
  {
    number: "11",
    href: ROUTES.ADMIN_RUNNERS,
    label: "Runners",
    // Under 72 characters, so the palette's length penalty never ranks it below a "@runner…" flag.
    body: "Find a runner by name or email, then suspend or message them.",
  },
  {
    number: "12",
    href: ROUTES.ADMIN_OVERVIEW,
    label: "Overview",
    body: "Weekly review — eight KPIs, a 30-day trend, and your latest decisions. Not the daily home.",
  },
  {
    number: "13",
    href: ROUTES.ADMIN_SALES,
    label: "Sales",
    body: "Where the money comes from — GMV, platform cut, refunds, top photographers and events.",
  },
  {
    number: "14",
    href: ROUTES.ADMIN_SECURITY,
    label: "Security",
    body: "Your own sign-in: set up two-step verification for this admin account.",
  },
];

export const ADMIN_SECTION_GUIDE: ReadonlyArray<SectionGuideEntry> =
  ADMIN_FLAGS_ENABLED
    ? ADMIN_SECTION_GUIDE_ALL
    : ADMIN_SECTION_GUIDE_ALL.filter((e) => e.href !== ROUTES.ADMIN_FLAGS);

export function getAdminSectionGuideByHref(
  href: string,
): SectionGuideEntry | undefined {
  return ADMIN_SECTION_GUIDE.find((entry) => entry.href === href);
}
