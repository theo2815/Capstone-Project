"use client";

import { useQuery } from "@tanstack/react-query";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { getMetadataSettings } from "@/lib/api-photographer";
import {
  fetchBrand,
  fetchPayoutAccounts,
  fetchSocials,
} from "@/lib/api-photographer-settings";

export const photographerSettingsKeys = {
  all: ["photographer", "settings"] as const,
  brand: ["photographer", "settings", "brand"] as const,
  socials: ["photographer", "settings", "socials"] as const,
  payouts: ["photographer", "settings", "payouts"] as const,
  metadata: ["photographer", "settings", "metadata"] as const,
};

const RETRY_DELAY_MS = 2_000;

const queryOptions = {
  retry: 2,
  retryDelay: RETRY_DELAY_MS,
} as const;

/** Server-owned photographer settings reads. */
export function usePhotographerSettingsQueries() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const querySuffix = [...session.key];

  return {
    session,
    brand: useQuery({
      queryKey: [...photographerSettingsKeys.brand, ...querySuffix],
      queryFn: fetchBrand,
      enabled: session.enabled,
      ...queryOptions,
    }),
    socials: useQuery({
      queryKey: [...photographerSettingsKeys.socials, ...querySuffix],
      queryFn: fetchSocials,
      enabled: session.enabled,
      ...queryOptions,
    }),
    payouts: useQuery({
      queryKey: [...photographerSettingsKeys.payouts, ...querySuffix],
      queryFn: fetchPayoutAccounts,
      enabled: session.enabled,
      ...queryOptions,
    }),
    metadata: useQuery({
      queryKey: [...photographerSettingsKeys.metadata, ...querySuffix],
      queryFn: ({ signal }) => getMetadataSettings(signal),
      enabled: session.enabled,
      ...queryOptions,
    }),
  };
}
