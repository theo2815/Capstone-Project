"use client";

import { useQuery } from "@tanstack/react-query";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { getSellerDetails } from "@/lib/api-seller-details";

// The photographer's own seller details (E1). Settings 09 and the verification
// status line share this one cached read.
export function useSellerDetails() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const queryKey = ["photographer", "seller-details", ...session.key];
  const query = useQuery({
    queryKey,
    queryFn: ({ signal }) => getSellerDetails(signal),
    enabled: session.enabled,
  });
  return { query, queryKey };
}
