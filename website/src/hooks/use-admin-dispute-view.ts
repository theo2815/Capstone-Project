"use client";

import { useAdminDispute } from "@/hooks/use-admin-disputes";
import { useAdminUsersData } from "@/lib/admin-users-data";

export function useAdminDisputeView(disputeId: string) {
  const query = useAdminDispute(disputeId);
  const { rows: userPool } = useAdminUsersData();
  const dispute = query.data;
  const photographer = userPool.find((user) => user.handle === dispute?.photographerHandle);
  return {
    ...query,
    data: dispute ? {
      dispute,
      photographerName: photographer?.brandName ?? photographer?.name ?? `@${dispute.photographerHandle}`,
    } : null,
  };
}
