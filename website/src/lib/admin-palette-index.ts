import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { ADMIN_ROUTES } from "@/components/admin/admin-rail";
import { ADMIN_QUEUES, queuePath } from "@/lib/admin-queues";
import { getAdminSectionGuideByHref } from "@/lib/admin-section-guide";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import { photographerHref, photographerStatus } from "@/lib/admin-photographer-view";
import {
  DISPUTE_REASON_LABEL,
  type Dispute,
} from "@/lib/admin-disputes";
import {
  flagReasonLabel,
  type Flag,
} from "@/lib/admin-flags";
import {
  ADMIN_PAYOUT_STATUS_LABEL,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";

// Phase 5 — admin command palette index. Pure functions only; no React,
// no store hooks. The palette supplies canonical dispute/flag/payout records and
// the remaining legacy user snapshots.
//
// Two surface contracts:
//   buildPaletteIndex(snapshots) → PaletteEntry[]   — flat, deduped list
//   fuzzyScore(query, entry)     → number | null   — null = no match
//
// Fuzzy scoring is character-skip with bonuses for word-start hits and
// contiguous runs (VSCode-style). Lowercase normalize once at index time
// so the matcher does no per-keystroke string allocation.

export type PaletteEntryKind =
  | "page"
  | "verification"
  | "dispute"
  | "flag"
  | "payout"
  | "photographer";

export interface PaletteEntry {
  id: string;
  kind: PaletteEntryKind;
  title: string;
  subtitle: string | null;
  /** Lowercase concatenation of every searchable string. */
  haystack: string;
  /** Full route string with query params if any. router.push(this) lands
   * the user at the right surface — for queue rows, that's the inbox with
   * ?type=...&row=<id> so the drawer auto-opens. */
  route: string;
}

export const PALETTE_KIND_LABEL: Record<PaletteEntryKind, string> = {
  page: "Pages",
  verification: "Verifications",
  dispute: "Disputes",
  flag: "Flags",
  payout: "Payouts",
  photographer: "Photographers",
};

/** Render order for grouped results — pages first, then queue verbs in
 *  the same order they appear in the inbox chip strip, then photographers. */
export const PALETTE_KIND_ORDER: ReadonlyArray<PaletteEntryKind> = [
  "page",
  "verification",
  "dispute",
  "flag",
  "payout",
  "photographer",
];

export interface PaletteSnapshots {
  /** Merged admin user pool — already includes server rows + optimistic
   *  backend rows. See hooks/use-admin-users.ts. */
  userPool: AdminUserRow[];
  disputes: ReadonlyArray<Dispute>;
  /** Server rows from useAdminFlags(); empty while loading or gated off. */
  flags: ReadonlyArray<Flag>;
  payouts: readonly AdminPayoutCycle[];
}

interface PageSeed {
  id: string;
  title: string;
  subtitle: string;
  route: string;
  /** Extra terms folded into haystack, so "dsp" or "gmv" still land on the page. */
  keywords: string[];
}

// Search words the titles and rail tooltips do not carry.
const KEYWORDS: Record<string, string[]> = {
  [ROUTES.ADMIN_OVERVIEW]: ["dashboard", "kpi", "metrics", "home"],
  [ROUTES.ADMIN_INBOX]: ["today", "stream"],
  [ROUTES.ADMIN_VERIFICATIONS]: ["approve", "reject", "pending", "vrf"],
  [ROUTES.ADMIN_DISPUTES]: ["refund", "claim", "dsp"],
  [ROUTES.ADMIN_FLAGS]: ["moderation", "report", "flg"],
  [ROUTES.ADMIN_PAYOUTS]: ["approve", "hold", "pay"],
  [ROUTES.ADMIN_RELEVANCE]: ["junk", "blurry", "copied", "wrong event", "alerts"],
  [ROUTES.ADMIN_PHOTOGRAPHERS]: ["directory", "users", "people"],
  [ROUTES.ADMIN_EVENTS]: ["registry", "races"],
  [ROUTES.ADMIN_SALES]: ["revenue", "money", "gmv", "earnings", "cut", "income"],
  [ROUTES.ADMIN_SECURITY]: ["two-step", "2fa", "mfa", "sign-in"],
  [`${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/new`]: ["create", "add", "takedown", "case"],
};

// Every rail page (ADMIN_ROUTES), the queues that live inside another page (seller details,
// payout reports), and the removal intake form.
const PAGE_SEEDS: ReadonlyArray<PageSeed> = [
  ...ADMIN_ROUTES.map((r) => ({ id: `page:${r.href.split("/").pop()}`, title: r.label, href: r.href })),
  ...ADMIN_QUEUES.filter((q) => q.href.includes("#")).map((q) => ({ id: `page:${q.inboxType}`, title: q.label, href: q.href })),
  { id: "page:photo-removal-new", title: "New removal request", href: `${ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS}/new` },
].map(({ id, title, href }) => ({
  id,
  title,
  subtitle: getAdminSectionGuideByHref(queuePath(href))?.body ?? "",
  route: href,
  keywords: KEYWORDS[href] ?? [],
}));

function buildPageEntries(): PaletteEntry[] {
  return PAGE_SEEDS.map((p) => ({
    id: p.id,
    kind: "page" as const,
    title: p.title,
    subtitle: p.subtitle,
    haystack: [p.title, p.subtitle, ...p.keywords].join(" ").toLowerCase(),
    route: p.route,
  }));
}

function buildVerificationEntries(
  userPool: AdminUserRow[],
): PaletteEntry[] {
  // Only photographers in pending or incomplete state appear — approved
  // photographers are reachable via the photographer entries below, and
  // suspended ones via the directory. Keeps the queue scope tight.
  const photographers = userPool.filter((u) => u.role === "PHOTOGRAPHER");
  const reviewable = photographers.filter(
    (u) =>
      (u.verificationStatus === "pending" ||
        u.verificationStatus === "incomplete") &&
      u.suspendedAt === null,
  );
  return reviewable.map((u) => {
    const status = u.verificationStatus === "pending" ? "Pending" : "Incomplete";
    const subjectLabel = u.brandName ?? u.name;
    const subtitle = `${status} · @${u.handle ?? "no-handle"} · ${u.city}`;
    return {
      id: `vrf:${u.userId}`,
      kind: "verification" as const,
      title: subjectLabel,
      subtitle,
      haystack: [
        subjectLabel,
        u.name,
        u.handle ?? "",
        u.email,
        status,
        u.region ?? "",
        u.city,
      ]
        .join(" ")
        .toLowerCase(),
      route: `${ROUTES.ADMIN_INBOX}?type=verifications&row=${encodeURIComponent(u.userId)}`,
    };
  });
}

function buildDisputeEntries(disputes: ReadonlyArray<Dispute>): PaletteEntry[] {
  return disputes.map((d) => {
    const reasonLabel = DISPUTE_REASON_LABEL[d.reason];
    const statusLabel = capitalize(d.status);
    const subtitle = `${statusLabel} · ${reasonLabel} · @${d.runnerHandle} ↔ @${d.photographerHandle}`;
    return {
      id: `dsp:${d.id}`,
      kind: "dispute" as const,
      title: d.id,
      subtitle,
      haystack: [
        d.id,
        d.orderId,
        d.runnerHandle,
        d.photographerHandle,
        statusLabel,
        reasonLabel,
        d.note,
      ]
        .join(" ")
        .toLowerCase(),
      route: `${ROUTES.ADMIN_INBOX}?type=disputes&row=${encodeURIComponent(d.id)}`,
    };
  });
}

function buildFlagEntries(flags: ReadonlyArray<Flag>): PaletteEntry[] {
  return flags.map((f) => {
    const reasonLabel = flagReasonLabel(f);
    const statusLabel = capitalize(f.status);
    const reporter = f.reportedBy === "system" ? "System" : `@${f.reportedBy}`;
    const subtitle = `${statusLabel} · ${reasonLabel} · ${reporter}`;
    return {
      id: `flg:${f.id}`,
      kind: "flag" as const,
      title: f.id,
      subtitle,
      haystack: [
        f.id,
        f.photographerHandle,
        f.reportedBy,
        statusLabel,
        reasonLabel,
        f.note,
      ]
        .join(" ")
        .toLowerCase(),
      route: `${ROUTES.ADMIN_INBOX}?type=flags&row=${encodeURIComponent(f.id)}`,
    };
  });
}

function buildPayoutEntries(
  payouts: readonly AdminPayoutCycle[],
): PaletteEntry[] {
  return payouts.map((p) => {
    const statusLabel = ADMIN_PAYOUT_STATUS_LABEL[p.status];
    const subjectLabel = p.brandName ?? p.photographerName;
    const subtitle = `${statusLabel} · ${subjectLabel} · ₱${p.amount.toLocaleString()} · ${p.itemCount} sales`;
    return {
      id: `pay:${p.id}`,
      kind: "payout" as const,
      title: p.id,
      subtitle,
      haystack: [
        p.id,
        subjectLabel,
        p.photographerName,
        p.handle ?? "",
        statusLabel,
        p.method,
      ]
        .join(" ")
        .toLowerCase(),
      route: `${ROUTES.ADMIN_INBOX}?type=payouts&row=${encodeURIComponent(p.id)}`,
    };
  });
}

function buildPhotographerEntries(
  userPool: AdminUserRow[],
): PaletteEntry[] {
  const photographers = userPool.filter((u) => u.role === "PHOTOGRAPHER");
  return photographers.map((u) => {
    const subjectLabel = u.brandName ?? u.name;
    const status = photographerStatus(u).label;
    const subtitle = `${status} · @${u.handle ?? "no-handle"} · ${u.city}`;
    const route = photographerHref(u);
    return {
      id: `phg:${u.userId}`,
      kind: "photographer" as const,
      title: subjectLabel,
      subtitle,
      haystack: [
        subjectLabel,
        u.name,
        u.handle ?? "",
        u.email,
        status,
        u.region ?? "",
        u.city,
      ]
        .join(" ")
        .toLowerCase(),
      route,
    };
  });
}

export function buildPaletteIndex(
  snapshots: PaletteSnapshots,
): PaletteEntry[] {
  return [
    ...buildPageEntries(),
    ...buildVerificationEntries(snapshots.userPool),
    ...buildDisputeEntries(snapshots.disputes),
    ...(ADMIN_FLAGS_ENABLED
      ? buildFlagEntries(snapshots.flags)
      : []),
    ...buildPayoutEntries(snapshots.payouts),
    ...buildPhotographerEntries(snapshots.userPool),
  ];
}

// ─── Fuzzy scorer ───────────────────────────────────────────────────────
//
// Character-skip matcher. Each query character must appear in haystack in
// order, not necessarily contiguous. Score is shaped so that:
//   - matches at word starts (after space/dash/underscore) score higher
//   - contiguous runs score higher than scattered hits
//   - shorter haystacks tie-break ahead of longer ones at equal score
// Returns null when any query character can't be found in order.

function isWordBoundary(prevChar: string | undefined): boolean {
  if (prevChar === undefined) return true;
  return prevChar === " " || prevChar === "-" || prevChar === "_" || prevChar === "·";
}

export function fuzzyScore(query: string, entry: PaletteEntry): number | null {
  const q = query.trim().toLowerCase();
  if (q.length === 0) return 0;
  const hay = entry.haystack;
  let qi = 0;
  let score = 0;
  let lastMatchIdx = -2;
  for (let hi = 0; hi < hay.length && qi < q.length; hi++) {
    if (hay[hi] !== q[qi]) continue;
    let bump = 1;
    if (isWordBoundary(hi === 0 ? undefined : hay[hi - 1])) bump += 3;
    if (hi === lastMatchIdx + 1) bump += 2;
    score += bump;
    lastMatchIdx = hi;
    qi++;
  }
  if (qi < q.length) return null;
  // Length penalty — prefer concise haystacks at equal raw score.
  return score - Math.floor(hay.length / 80);
}

function capitalize(s: string): string {
  if (s.length === 0) return s;
  return s[0].toUpperCase() + s.slice(1);
}
