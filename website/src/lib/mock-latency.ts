"use client";

import { useEffect, useState } from "react";

// Synthetic latency seam for demoing skeleton loading states before the
// backend is wired. With MOCK_LATENCY_MS = 0 (the dev default) this is a
// pure pass-through — useMockLatency returns the value immediately and
// isLoading is false on first render, so consumers pay no overhead and
// skeletons never flash.
//
// To stress-test skeletons (or record a demo), bump MOCK_LATENCY_MS to a
// value like 800 here. Every consumer using useMockLatency will then show
// its skeleton state until the timer resolves. Revert to 0 before commit.
//
// TODO(backend): once real fetches land, replace useMockLatency call sites
// with the actual data-fetching hook (e.g. SWR / React Query). The
// `{ data, isLoading }` shape was chosen to match those libraries so the
// migration is mostly a search-and-replace.

export const MOCK_LATENCY_MS = 0;

export interface MockLatencyResult<T> {
  data: T | null;
  isLoading: boolean;
}

export function useMockLatency<T>(value: T): MockLatencyResult<T> {
  // Fast path: zero latency means resolved-on-mount. Skip the effect entirely
  // so we never schedule a timer or flash a skeleton frame.
  const [resolved, setResolved] = useState<T | null>(
    MOCK_LATENCY_MS > 0 ? null : value,
  );

  useEffect(() => {
    if (MOCK_LATENCY_MS === 0) {
      // Value-identity changes (parent re-rendered with a new reference)
      // still need to propagate — keep `resolved` in sync with the latest.
      setResolved(value);
      return;
    }
    setResolved(null);
    const timer = setTimeout(() => setResolved(value), MOCK_LATENCY_MS);
    return () => clearTimeout(timer);
  }, [value]);

  return { data: resolved, isLoading: resolved === null };
}
