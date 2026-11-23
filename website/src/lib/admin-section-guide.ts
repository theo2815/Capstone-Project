import type { SectionGuideEntry } from "@/lib/section-guide";
import type { AdminQueueKey } from "@/lib/admin-queues";
import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";

// Mirror of `SECTION_GUIDE` for the photographer dashboard, but for the
// admin workspace. Single source of truth for "what each rail tab does":
// `body` feeds the rail tips and the palette subtitles; `intro`, `instead`
// and `terms` are shown on the page itself by <AdminSectionIntro>.
//
// Entries follow the rail's order and numbers (admin-rail.tsx): Inbox,
// then the queues in ADMIN_QUEUES order, then Catalog, Reports and Account.

export interface AdminSectionGuideEntry extends SectionGuideEntry {
  /** What is on the page and what the admin does there. */
  readonly intro: string;
  /** The look-alike pages, rendered as "{when}, use {label}." */
  readonly instead?: ReadonlyArray<{ when: string; label: string; href: string }>;
  /** Words that need a definition, rendered as "{term}: {meaning}". */
  readonly terms?: ReadonlyArray<{ term: string; meaning: string }>;
}

const ADMIN_SECTION_GUIDE_ALL: ReadonlyArray<AdminSectionGuideEntry> = [
  {
    number: "01",
    href: ROUTES.ADMIN_INBOX,
    label: "Inbox",
    body: "Your morning landing: every queue waiting on an admin, one chip at a time.",
    intro:
      "Everything that is waiting for an admin, one queue at a time. Start with the queue marked by the green dot: it is the first one with work. A number is how many items are waiting for a decision right now.",
    instead: [{ when: "For totals and reports", label: "Overview", href: ROUTES.ADMIN_OVERVIEW }],
    terms: [{ term: "Queue", meaning: "a list of items of one kind that need a decision." }],
  },
  {
    number: "02",
    href: ROUTES.ADMIN_DISPUTES,
    label: "Disputes",
    body: "Refund requests and complaints from runners. Resolve before they escalate.",
    intro:
      "Refund requests and complaints from runners about photos they bought. Read the facts on each one, then approve the refund or deny it with a reason the runner will read.",
    instead: [{ when: "For a report about a race or a photographer", label: "Flags", href: ROUTES.ADMIN_FLAGS }],
    terms: [
      { term: "Overdue", meaning: "past the due date shown on the dispute." },
      { term: "Refund run", meaning: "approved QR Ph refunds that you still have to pay by hand in PayMongo." },
      { term: "PayMongo", meaning: "the payment service that collects runners' payments and sends refunds." },
      { term: "QR Ph", meaning: "paying by scanning a QR code with a bank or e-wallet app." },
    ],
  },
  {
    number: "03",
    href: ROUTES.ADMIN_FLAGS,
    label: "Flags",
    body: "Reports about a race or a photographer, and things the automatic checks marked.",
    intro:
      "Reports about a race or a photographer, sent by runners and photographers, and things QuickPitik's automatic checks marked. Read each one and close it, or act on it from the page it points to.",
    instead: [
      { when: "For a whole upload the checker doubts", label: "Photo Relevance", href: ROUTES.ADMIN_RELEVANCE },
      {
        when: "For someone asking to have photos of them taken down",
        label: "Photo removals",
        href: ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS,
      },
    ],
    terms: [
      {
        term: "Automatic check",
        meaning: "a rule the system runs by itself, for example on a new profile picture or a changed payout name.",
      },
    ],
  },
  {
    number: "04",
    href: ROUTES.ADMIN_VERIFICATIONS,
    label: "Verifications",
    body: "Profiles to send back or approve, and the ID checks that need a person.",
    intro:
      "Two lists. Profiles: most go live by themselves once the ID check passes; here you send one back for fixes, or approve one that was sent back before. Seller details: the ID checks that need a person.",
    instead: [
      { when: "To look up, suspend or message any photographer", label: "Photographers", href: ROUTES.ADMIN_PHOTOGRAPHERS },
    ],
    terms: [
      {
        term: "Seller details",
        meaning: "the legal name, ID and payout account a photographer sends before they may upload.",
      },
      { term: "Held", meaning: "sent back once already; after the photographer fixes it, it waits for your approval." },
    ],
  },
  {
    number: "05",
    href: ROUTES.ADMIN_EVENT_REQUESTS,
    label: "Event requests",
    body: "Photographer-created events waiting on a decision, and pricing changes parked on live ones. Approve, or send back with a reason.",
    intro:
      "Events made by photographers, and price changes on events that are already live. Nothing here is public or changed until you approve it.",
    instead: [{ when: "To add or edit an event yourself", label: "Events", href: ROUTES.ADMIN_EVENTS }],
  },
  {
    number: "06",
    href: ROUTES.ADMIN_PAYOUTS,
    label: "Payouts",
    body: "Payout requests across all photographers. Mark one paid or hold it.",
    intro:
      "Photographers asking to be paid. Three steps: 1. Approve. This sends no money. 2. Send the money yourself to the account shown. 3. Mark it paid and type the bank or wallet reference. Reports from photographers about a payout that went wrong are lower on this page.",
    terms: [
      { term: "Hold", meaning: "pause a payout; the photographer is told why." },
      { term: "Mark processing", meaning: "you have started the transfer." },
    ],
  },
  {
    number: "07",
    href: ROUTES.ADMIN_RELEVANCE,
    label: "Photo Relevance",
    body: "Batches checked after upload for junk, wrong-event and copied photos. Read each verdict, its reason and its sample photos.",
    intro:
      "After each upload, an automatic checker looks at the batch for junk, photos from another event and copied photos. It can take photos off sale by itself. Your part: judge the batches it could not, decide appeals, and undo its mistakes.",
    instead: [
      { when: "For a race or photographer someone reported", label: "Flags", href: ROUTES.ADMIN_FLAGS },
      {
        when: "For someone asking to have photos of them taken down",
        label: "Photo removals",
        href: ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS,
      },
    ],
    terms: [
      { term: "Batch", meaning: "all the photos one photographer uploaded to one event." },
      { term: "Case", meaning: "one batch the checker doubted." },
      { term: "Would have hidden", meaning: "recorded only; nothing was hidden." },
      { term: "Can't tell", meaning: "the checker could not decide, so a person must." },
    ],
  },
  {
    number: "08",
    href: ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS,
    label: "Photo removals",
    body: "Runner requests and support requests. Review what was hidden or suggested, find their photos across every photographer in one event and verify which to hide.",
    intro:
      "Requests to take photos down, from runners and through support. You confirm which photos are meant, each photographer reviews their own, then you decide photo by photo. A removed photo is deleted for good.",
    instead: [
      { when: "For a doubtful upload", label: "Photo Relevance", href: ROUTES.ADMIN_RELEVANCE },
      { when: "For a reported race or photographer", label: "Flags", href: ROUTES.ADMIN_FLAGS },
    ],
    terms: [
      { term: "Retain", meaning: "keep the photo and put it back on sale." },
      { term: "Retain restricted", meaning: "keep the photo but leave it hidden, with a date to review it again." },
      { term: "Guest buyer", meaning: "someone who bought without an account; QuickPitik cannot message them." },
    ],
  },
  {
    number: "09",
    href: ROUTES.ADMIN_EVENTS,
    label: "Events",
    body: "Every event. Add one, edit its details and price, or hide it.",
    intro:
      "Every event on QuickPitik. Add one, edit its details and price, or hide it. An event moves between the tabs by itself as its date passes.",
    instead: [{ when: "To approve an event a photographer made", label: "Event requests", href: ROUTES.ADMIN_EVENT_REQUESTS }],
    terms: [
      { term: "Live now", meaning: "race day and the few days right after it." },
      { term: "Recent", meaning: "the weeks after that, while the photos are on sale." },
      { term: "Archive", meaning: "older events." },
    ],
  },
  {
    number: "10",
    href: ROUTES.ADMIN_PHOTOGRAPHERS,
    label: "Photographers",
    body: "Directory of approved + pending photographers. Drill in for history and brand checks.",
    intro: "Every photographer. Find one, then open them to suspend, reset or message them.",
    instead: [{ when: "To decide a new profile or an ID check", label: "Verifications", href: ROUTES.ADMIN_VERIFICATIONS }],
    terms: [
      { term: "Pending", meaning: "sent their profile and is waiting for the ID check, or for you." },
      { term: "Incomplete", meaning: "still filling in their settings, or sent back for fixes." },
      { term: "Suspended", meaning: "cannot sign in; their photos are off sale." },
    ],
  },
  {
    number: "11",
    href: ROUTES.ADMIN_RUNNERS,
    label: "Runners",
    // Under 72 characters, so the palette's length penalty never ranks it below a "@runner…" flag.
    body: "Find a runner by name or email, then suspend or message them.",
    intro:
      "Every runner. Find one by name or email, then suspend or message them. Look at their disputes before you suspend.",
    instead: [{ when: "For a runner's refund request", label: "Disputes", href: ROUTES.ADMIN_DISPUTES }],
  },
  {
    number: "12",
    href: ROUTES.ADMIN_OVERVIEW,
    label: "Overview",
    body: "A report: totals right now, this week's decisions and the last 30 days. Today's work is in the Inbox.",
    intro:
      "A report, not a to-do list: totals as they stand right now, the decisions made this week, and the last 30 days.",
    instead: [{ when: "For today's work", label: "Inbox", href: ROUTES.ADMIN_INBOX }],
  },
  {
    number: "13",
    href: ROUTES.ADMIN_SALES,
    label: "Sales",
    body: "Where the money comes from: total sales, the platform cut, refunds, top photographers and events.",
    intro:
      "Where the money comes from: what runners paid, QuickPitik's share, and refunds. Each section says which period it covers.",
    terms: [
      { term: "Total sales", meaning: "everything runners paid, before the photographer's share." },
      { term: "Platform cut", meaning: "QuickPitik's share of those sales." },
      {
        term: "Estimate",
        meaning: "worked out from an event's photo count because no sales figure has arrived for it yet.",
      },
    ],
  },
  {
    number: "14",
    href: ROUTES.ADMIN_SECURITY,
    label: "Security",
    body: "Your own sign-in: set up two-step verification for this admin account.",
    intro:
      "Your own sign-in. Two-step verification asks for a code from your phone each time you sign in, so a stolen password is not enough. Turn it on if you approve payouts or refunds.",
    terms: [
      {
        term: "Authenticator app",
        meaning: "a free phone app, for example Google Authenticator, that shows a new 6-digit code about every 30 seconds.",
      },
      { term: "Recovery codes", meaning: "one-time codes for when you do not have your phone." },
    ],
  },
];

// The Flags gate drops the Flags entry and every "use Flags instead" pointer.
export const ADMIN_SECTION_GUIDE: ReadonlyArray<AdminSectionGuideEntry> =
  ADMIN_FLAGS_ENABLED
    ? ADMIN_SECTION_GUIDE_ALL
    : ADMIN_SECTION_GUIDE_ALL.filter((e) => e.href !== ROUTES.ADMIN_FLAGS).map((e) => ({
        ...e,
        instead: e.instead?.filter((i) => i.href !== ROUTES.ADMIN_FLAGS),
      }));

export function getAdminSectionGuideByHref(
  href: string,
): AdminSectionGuideEntry | undefined {
  return ADMIN_SECTION_GUIDE.find((entry) => entry.href === href);
}

/** The sentence under the Inbox chips for the open queue: what it holds and what to do. */
export const ADMIN_QUEUE_LINES: Record<AdminQueueKey, string> = {
  disputes:
    "A runner asked for a refund or reported a problem with a photo they bought. Open one, read the facts, then approve the refund or deny it with a reason.",
  flags:
    "A race or a photographer that someone reported, or that an automatic check marked. Open one and decide whether anything needs doing.",
  verifications:
    "Photographer profiles. Most go live by themselves when the ID check passes. Send one back if it needs fixes, or approve one that was sent back before.",
  sellerDetails:
    "ID checks the system could not pass by itself. Compare the ID, the live selfie and the payout name, then approve or reject.",
  eventRequests:
    "Events made by photographers, and price changes on live ones. Nothing here is public until you approve it.",
  payouts:
    "Photographers asking to be paid. Approving sends no money: you send it yourself, then mark the payout paid.",
  payoutReports:
    "A photographer says a payout went wrong. Reply so they know you are on it, then resolve the report once it is fixed.",
  relevance:
    "Uploads the automatic checker could not judge, and photographers' appeals. Look at the sample photos and decide.",
  removals:
    "Someone asked for photos of them to be taken down. Confirm which photos, wait for the photographers' answers, then decide photo by photo.",
};
