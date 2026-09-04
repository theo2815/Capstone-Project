"use client";

import { ADMIN_FLAGS_ENABLED } from "@/lib/constants";
import { cn } from "@/lib/utils";

export type InboxQueueType =
  | "verifications"
  | "events"
  | "disputes"
  | "flags"
  | "payouts";

const INBOX_QUEUE_TYPES_ALL: readonly InboxQueueType[] = [
  "verifications",
  "events",
  "disputes",
  "flags",
  "payouts",
] as const;

// Flags chip is hidden when the v1 feature gate is off — see constants.ts.
export const INBOX_QUEUE_TYPES: readonly InboxQueueType[] = ADMIN_FLAGS_ENABLED
  ? INBOX_QUEUE_TYPES_ALL
  : INBOX_QUEUE_TYPES_ALL.filter((t) => t !== "flags");

const LABEL: Record<InboxQueueType, string> = {
  verifications: "Verifications",
  events: "Event requests",
  disputes: "Disputes",
  flags: "Flags",
  payouts: "Payouts",
};

interface AdminInboxFilterChipsProps {
  value: InboxQueueType;
  onChange: (next: InboxQueueType) => void;
}

// Inbox queue-type filter chips. Sits between the KPI strip and the
// active queue body. Active chip is ink-filled; inactive chips are
// bone-bordered. Counts are deliberately omitted — the KPI strip
// directly above already carries them, and doubling up adds visual
// noise. Comfort-floor mono type, wraps on small screens.
export function AdminInboxFilterChips({
  value,
  onChange,
}: AdminInboxFilterChipsProps) {
  return (
    <div
      role="tablist"
      aria-label="Inbox queue filter"
      className="flex flex-wrap gap-2 pt-6 md:pt-8"
    >
      {INBOX_QUEUE_TYPES.map((type) => {
        const active = value === type;
        return (
          <button
            key={type}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => onChange(type)}
            className={cn(
              "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-4 py-2 transition-colors",
              active
                ? "bg-ink text-bone border border-ink"
                : "bg-bone text-slate border border-line hover:text-ink hover:border-ink",
            )}
          >
            {LABEL[type]}
          </button>
        );
      })}
    </div>
  );
}

export function isInboxQueueType(raw: string): raw is InboxQueueType {
  return (INBOX_QUEUE_TYPES as readonly string[]).includes(raw);
}
