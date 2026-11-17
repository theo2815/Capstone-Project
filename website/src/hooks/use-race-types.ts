"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchRaceTypes, type RaceType } from "@/lib/api-photographer";

// `GET /api/v1/race-types` (AA-R1): the backend owns the race-type codes and
// labels; clients store codes only. Public, `Cache-Control: max-age=300`.
// `selfServeEnabled` mirrors SELF_SERVE_RACES_ENABLED: off = today's form.
export function useRaceTypes(): { selfServeEnabled: boolean; types: readonly RaceType[] } {
  const query = useQuery({
    queryKey: ["race-types"],
    queryFn: fetchRaceTypes,
    staleTime: 5 * 60 * 1000,
  });
  return {
    selfServeEnabled: query.data?.selfServeEnabled === true,
    types: query.data?.types ?? [],
  };
}
