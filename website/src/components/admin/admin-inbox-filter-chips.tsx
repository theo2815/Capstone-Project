"use client";

import { useRouter } from "next/navigation";
import { AdminStatusTabs } from "@/components/admin/admin-status-tabs";
import { ADMIN_QUEUES, useAdminQueueCounts, type AdminQueue } from "@/lib/admin-queues";
import { ROUTES } from "@/lib/constants";

export type InboxQueueType = AdminQueue["inboxType"];

// Inbox queue chips: every queue in ADMIN_QUEUES, in its order, with its KPI count.
export function AdminInboxFilterChips({ value }: { value: InboxQueueType }) {
  const router = useRouter();
  const { byKey } = useAdminQueueCounts();
  return (
    <AdminStatusTabs
      label="Inbox queue filter"
      tabs={ADMIN_QUEUES.map((q) => ({ key: q.inboxType, label: q.label, count: byKey[q.key] ?? undefined }))}
      value={value}
      // The Inbox owns only `type`: a switch drops the last list's own params (row, status, offset…).
      onChange={(next) => router.replace(`${ROUTES.ADMIN_INBOX}?type=${next}`, { scroll: false })}
    />
  );
}

export function isInboxQueueType(raw: string): raw is InboxQueueType {
  return ADMIN_QUEUES.some((q) => q.inboxType === raw);
}
