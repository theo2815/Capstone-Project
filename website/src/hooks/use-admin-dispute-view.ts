"use client";

import { useAdminDispute } from "@/hooks/use-admin-disputes";
import { useAdminUsers, EMPTY_ADMIN_USERS } from "@/hooks/use-admin-users";

export function useAdminDisputeView(disputeId: string) {
  const query = useAdminDispute(disputeId);
  const userPool = useAdminUsers().data ?? EMPTY_ADMIN_USERS;
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
