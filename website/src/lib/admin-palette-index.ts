import { ROUTES, ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { type AdminUserRow } from "@/lib/admin-user-registry";
import {
  ADMIN_DISPUTES,
  DISPUTE_REASON_LABEL,
  type Dispute,
} from "@/lib/admin-disputes";
import {
  flagReasonLabel,
  type Flag,
} from "@/lib/admin-flags";
import {
  ADMIN_PAYOUT_SEED,
  ADMIN_PAYOUT_STATUS_LABEL,
  type AdminPayoutCycle,
} from "@/lib/admin-payouts";
import { mergeDispute } from "@/store/admin-dispute-store";
import { mergePayout } from "@/store/admin-payout-store";

// Phase 5 — admin command palette index. Pure functions only; no React,
// no store hooks. The palette component passes in current store snapshots
// (overrides maps) so the index reflects in-flight admin actions instead
// of stale seeds.
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
   *  overrides + submissions. See lib/admin-users-data.ts. */
  userPool: AdminUserRow[];
  disputeOverrides: Record<string, Partial<Dispute>>;
  /** Server rows from useAdminFlags(); empty while loading or gated off. */
  flags: ReadonlyArray<Flag>;
  payoutOverrides: Record<string, Partial<AdminPayoutCycle>>;
}

interface PageSeed {
  id: string;
  title: string;
  subtitle: string;
  route: string;
  /** Extra terms folded into haystack — e.g. "inbox" for verifications so
   *  typing "inbox verifications" or "vrf" both land on the same entry. */
  keywords: string[];
}

const PAGE_SEEDS: ReadonlyArray<PageSeed> = [
  {
    id: "page:overview",
    title: "Overview",
    subtitle: "Admin dashboard · KPIs and platform health",
    route: ROUTES.ADMIN_OVERVIEW,
    keywords: ["dashboard", "kpi", "metrics", "home"],
  },
  {
    id: "page:inbox",
    title: "Inbox",
    subtitle: "Today · all four queues in one stream",
    route: ROUTES.ADMIN_INBOX,
    keywords: ["today", "stream"],
  },
  {
    id: "page:verifications",
    title: "Verifications",
    subtitle: "Photographer onboarding decisions",
    route: `${ROUTES.ADMIN_INBOX}?type=verifications`,
    keywords: ["inbox", "approve", "reject", "pending", "vrf"],
  },
  {
    id: "page:verifications-direct",
    title: "Verifications (focus mode)",
    subtitle: "Standalone queue page with full hero",
    route: ROUTES.ADMIN_VERIFICATIONS,
    keywords: ["focus"],
  },
  {
    id: "page:disputes",
    title: "Disputes",
    subtitle: "Refund + delivery claims",
    route: `${ROUTES.ADMIN_INBOX}?type=disputes`,
    keywords: ["inbox", "refund", "claim", "dsp"],
  },
  ...(ADMIN_FLAGS_ENABLED
    ? [
        {
          id: "page:flags",
          title: "Flags",
          subtitle: "Content moderation queue",
          route: `${ROUTES.ADMIN_INBOX}?type=flags`,
          keywords: ["inbox", "moderation", "report", "flg"],
        } as PageSeed,
      ]
    : []),
  {
    id: "page:payouts",
    title: "Payouts",
    subtitle: "Photographer payout requests",
    route: `${ROUTES.ADMIN_INBOX}?type=payouts`,
    keywords: ["inbox", "approve", "hold", "pay"],
  },
  {
    id: "page:photographers",
    title: "Photographers",
    subtitle: "Directory · all profiles by status",
    route: ROUTES.ADMIN_PHOTOGRAPHERS,
    keywords: ["directory", "users", "people"],
  },
  {
    id: "page:events",
    title: "Events",
    subtitle: "Admin-curated event registry",
    route: ROUTES.ADMIN_EVENTS,
    keywords: ["registry", "races"],
  },
  {
    id: "page:events-new",
    title: "Events · New",
    subtitle: "Create a new event",
    route: `${ROUTES.ADMIN_EVENTS}/new`,
    keywords: ["create", "add", "event"],
  },
  {
    id: "page:sales",
    title: "Sales",
    subtitle: "Revenue intelligence · GMV, platform cut, top performers",
    route: ROUTES.ADMIN_SALES,
    keywords: ["revenue", "money", "gmv", "earnings", "cut", "income"],
  },
];

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

function buildDisputeEntries(
  disputeOverrides: Record<string, Partial<Dispute>>,
): PaletteEntry[] {
  const merged = ADMIN_DISPUTES.map((d) =>
    mergeDispute(d, disputeOverrides[d.id]),
  );
  return merged.map((d) => {
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
    const reasonLabel = flagReasonLabel(f.reason);
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
  payoutOverrides: Record<string, Partial<AdminPayoutCycle>>,
): PaletteEntry[] {
  const merged = ADMIN_PAYOUT_SEED.map((p) =>
    mergePayout(p, payoutOverrides[p.id]),
  );
  return merged.map((p) => {
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
    const isSuspended = u.suspendedAt !== null;
    const status = isSuspended
      ? "Suspended"
      : u.verificationStatus === "approved"
        ? "Approved"
        : u.verificationStatus === "pending"
          ? "Pending"
          : "Incomplete";
    const subtitle = `${status} · @${u.handle ?? "no-handle"} · ${u.city}`;
    // Photographers without a handle have no detail page; fall back to the
    // directory and let click semantics handle the no-op there.
    const route = u.handle
      ? `${ROUTES.ADMIN_PHOTOGRAPHERS}/${u.handle}`
      : ROUTES.ADMIN_PHOTOGRAPHERS;
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
    ...buildDisputeEntries(snapshots.disputeOverrides),
    ...(ADMIN_FLAGS_ENABLED
      ? buildFlagEntries(snapshots.flags)
      : []),
    ...buildPayoutEntries(snapshots.payoutOverrides),
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
