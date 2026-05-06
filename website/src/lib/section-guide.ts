import { ROUTES } from "@/lib/constants";

// Single source of copy for the photographer-dashboard sidebar guide.
// Consumed by:
//   - SetupJourney's <SectionsAtAGlance> panel (first-timer overview)
//   - DashboardRail per-item tip toggles (always available)
// Edit copy here so both surfaces stay in sync.

export interface SectionGuideEntry {
  readonly number: string;
  readonly href: string;
  readonly label: string;
  readonly body: string;
}

export const SECTION_GUIDE: ReadonlyArray<SectionGuideEntry> = [
  {
    number: "01",
    href: ROUTES.DASHBOARD,
    label: "Overview",
    body: "Today at a glance — what's live, what's queued, what's owed. The control room you're looking at right now.",
  },
  {
    number: "02",
    href: ROUTES.DASHBOARD_UPLOAD,
    label: "Upload photos",
    body: "Pick a race and lift photos to it. Live races accept uploads now; upcoming ones unlock on race day.",
  },
  {
    number: "03",
    href: ROUTES.DASHBOARD_EVENTS,
    label: "Events",
    body: "The races you've covered. Open any to grab the public gallery link and share it to your followers.",
  },
  {
    number: "04",
    href: ROUTES.DASHBOARD_EARNINGS,
    label: "Earnings",
    body: "How much you've kept, sale by sale. The trend chart shows where the work is paying off.",
  },
  {
    number: "05",
    href: ROUTES.DASHBOARD_BILLING,
    label: "Billing",
    body: "When the money lands and which account it lands in. Payouts cycle weekly via GCash.",
  },
  {
    number: "06",
    href: ROUTES.DASHBOARD_SETTINGS,
    label: "Settings",
    body: "Your brand, your handle, your payout account. Finish this once and uploads unlock for every race.",
  },
];

export function getSectionGuideByHref(
  href: string,
): SectionGuideEntry | undefined {
  return SECTION_GUIDE.find((s) => s.href === href);
}
