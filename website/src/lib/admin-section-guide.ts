import type { SectionGuideEntry } from "@/lib/section-guide";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// Mirror of `SECTION_GUIDE` for the photographer dashboard, but for the
// admin workspace. Single source of truth for "what each rail tab does"
// — consumed by `<AdminRail>` per-row tooltips and (once Phase 2 lands)
// by an admin first-timer overview panel.
//
// Inbox-led ordering (Phase 1 admin redesign): the daily home is the
// universal queue at /admin/inbox; verifications, disputes, flags, payouts
// each keep their own routes for deep-linking and focused-mode use, but
// the rail surfaces them in priority order. Overview (KPI grid + 30-day
// trend + decisions log) demotes to last entry — the weekly-review tab.

const ADMIN_SECTION_GUIDE_ALL: ReadonlyArray<SectionGuideEntry> = [
  {
    number: "01",
    href: ROUTES.ADMIN_INBOX,
    label: "Inbox",
    body: "Your morning landing. Pending verifications by default — disputes and payouts arrive as filter chips.",
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
    href: ROUTES.ADMIN_PAYOUTS,
    label: "Payouts",
    body: "Weekly cycle status across all photographers. Mark a cycle paid or hold.",
  },
  {
    number: "05",
    href: ROUTES.ADMIN_EVENTS,
    label: "Events",
    body: "Every race on the platform. Edit lifecycle state, archive, or unhide.",
  },
  {
    number: "06",
    href: ROUTES.ADMIN_PHOTOGRAPHERS,
    label: "Photographers",
    body: "Directory of approved + pending photographers. Drill in for history and brand checks.",
  },
  {
    number: "07",
    href: ROUTES.ADMIN_OVERVIEW,
    label: "Overview",
    body: "Weekly review — eight KPIs, a 30-day trend, and your latest decisions. Not the daily home.",
  },
  {
    number: "08",
    href: ROUTES.ADMIN_SALES,
    label: "Sales",
    body: "Where the money comes from — GMV, platform cut, refunds, top photographers and events.",
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
