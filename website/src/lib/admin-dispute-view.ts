"use client";

import { useMemo } from "react";
import {
  ADMIN_DISPUTES,
  type Dispute,
} from "@/lib/admin-disputes";
import {
  useAdminDisputeStore,
  type DisputeLogEntry,
  mergeDispute,
} from "@/store/admin-dispute-store";
import { useAdminUsersData } from "@/lib/admin-users-data";

// Reactive merged view used by the dispute detail page.
//
// Returns:
//   - dispute (effective seed + override) or null if id doesn't resolve
//   - photographerName: resolved via admin-user-registry merged with the
//     user-store overrides (so a force-edited brand reflects here).
//     Falls back to the handle string.
//   - decisions: log entries for THIS dispute, newest first.

export interface AdminDisputeView {
  dispute: Dispute;
  photographerName: string;
  decisions: DisputeLogEntry[];
}

export function useAdminDisputeView(
  disputeId: string,
): AdminDisputeView | null {
  const disputeOverrides = useAdminDisputeStore((s) => s.overrides);
  const log = useAdminDisputeStore((s) => s.log);
  const { rows: userPool } = useAdminUsersData();

  return useMemo(() => {
    const base = ADMIN_DISPUTES.find((d) => d.id === disputeId);
    if (!base) return null;
    const dispute = mergeDispute(base, disputeOverrides[disputeId]);

    const photogRow = userPool.find(
      (u) => u.handle === dispute.photographerHandle,
    );
    const photographerName =
      photogRow?.brandName ?? photogRow?.name ?? `@${dispute.photographerHandle}`;

    const decisions = log
      .filter((e) => e.disputeId === disputeId)
      .slice(0, 20);

    return { dispute, photographerName, decisions };
  }, [disputeId, disputeOverrides, userPool, log]);
}
