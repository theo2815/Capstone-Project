"use client";

import { useRouter } from "next/navigation";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { ADMIN_QUEUES, useAdminAttentionTarget, useAdminQueueCounts, type AdminQueue } from "@/lib/admin-queues";
import { ROUTES } from "@/lib/constants";

export type InboxQueueType = AdminQueue["inboxType"];

// Inbox queue chips: every queue in ADMIN_QUEUES, in its order, with its KPI count.
// They are the Inbox's only counts; the attention queue's chip carries the fresh dot the rail lights.
export function AdminInboxFilterChips({ value }: { value: InboxQueueType }) {
  const router = useRouter();
  const { byKey } = useAdminQueueCounts();
  const attention = useAdminAttentionTarget();
  return (
    <AdminStatusTabs
      label="Inbox queue filter"
      tabs={ADMIN_QUEUES.map((q) => ({
        key: q.inboxType,
        label: q.label,
        count: byKey[q.key] ?? undefined,
        dot: attention?.key === q.key,
      }))}
      value={value}
      scroll
      countLabel="waiting"
      // The Inbox's first screen belongs to the queue: the chips sit closer to the header.
      className="mt-4 md:mt-6"
      // The Inbox owns only `type`: a switch drops the last list's own params (row, status, offset…).
      onChange={(next) => router.replace(`${ROUTES.ADMIN_INBOX}?type=${next}`, { scroll: false })}
    />
  );
}
