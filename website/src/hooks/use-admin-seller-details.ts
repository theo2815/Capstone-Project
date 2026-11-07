"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  approveSellerDetails,
  getSellerDetailsRecord,
  listSellerDetails,
  rejectSellerDetails,
} from "@/lib/api-seller-details";
import { useAuthStore } from "@/store/auth-store";

const sellerKeys = {
  owner: (owner: string | null) => ["admin", "seller-details", owner] as const,
  list: (owner: string | null, status: "submitted" | null, limit: number) =>
    [...sellerKeys.owner(owner), "list", status, limit] as const,
  record: (owner: string | null, userId: string | null) => [...sellerKeys.owner(owner), "record", userId] as const,
};

function useAdminOwner() {
  const user = useAuthStore((s) => s.user);
  return { owner: user?.id ?? null, enabled: user?.role === "ADMIN" };
}

export function useAdminSellerDetailsList(status: "submitted" | null, limit: number) {
  const { owner, enabled } = useAdminOwner();
  return useQuery({
    queryKey: sellerKeys.list(owner, status, limit),
    queryFn: ({ signal }) => listSellerDetails(status, limit, signal),
    enabled,
    placeholderData: keepPreviousData,
  });
}

// Every fetch makes new 5-minute links and logs a "viewed" row, so the record
// is read only while the drawer is open and never refetched behind the admin's back.
export function useAdminSellerDetailsRecord(userId: string | null) {
  const { owner, enabled } = useAdminOwner();
  return useQuery({
    queryKey: sellerKeys.record(owner, userId),
    queryFn: ({ signal }) => getSellerDetailsRecord(userId!, signal),
    enabled: enabled && !!userId,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
    gcTime: 0,
    staleTime: 4 * 60_000,
    retry: false,
  });
}

export function useAdminSellerDetailsActions() {
  const qc = useQueryClient();
  const { owner } = useAdminOwner();
  const refreshLists = () => qc.invalidateQueries({ queryKey: [...sellerKeys.owner(owner), "list"] });
  const approve = useMutation({
    mutationFn: ({ userId, revision }: { userId: string; revision: number }) => approveSellerDetails(userId, revision),
    onSuccess: refreshLists,
  });
  const reject = useMutation({
    mutationFn: ({ userId, revision, reason }: { userId: string; revision: number; reason: string }) =>
      rejectSellerDetails(userId, revision, reason),
    onSuccess: refreshLists,
  });
  return { approve, reject, isPending: approve.isPending || reject.isPending };
}
