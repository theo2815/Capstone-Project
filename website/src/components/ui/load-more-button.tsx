"use client";

import { useEffect, useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { cn } from "@/lib/utils";

interface LoadMoreButtonProps {
  shown: number;
  total: number;
  increment: number;
  onLoadMore: () => void;
  /** Override the button label. Default: "Load N more". */
  label?: string;
  /** Override the count line above the button. Default: "Showing S of T". */
  countLabel?: string;
  /** Override the terminal kicker shown when shown >= total. Default: "All N loaded". */
  terminalLabel?: string;
  /** Optional caption appended after the count (e.g. " · Bib B-4082"). */
  countSuffix?: string;
  /** Disable button (e.g. while a parent is fetching). */
  isLoading?: boolean;
  className?: string;
}

// Quiet Studio load-more primitive used across runner + photographer surfaces.
// Renders nothing when total === 0 (let the parent's empty-state component speak).
// When shown >= total > 0, renders a single soft-tone terminal kicker instead of the button,
// unless the list was complete from the start (then nothing: there was nothing to load).
// One-fresh-per-viewport: the button uses ink, never fresh — load-more is utility chrome.
export function LoadMoreButton({
  shown,
  total,
  increment,
  onLoadMore,
  label,
  countLabel,
  terminalLabel,
  countSuffix,
  isLoading = false,
  className,
}: LoadMoreButtonProps) {
  // A list that was complete from the start never offered "Load more", so
  // "All 2 loaded" under two rows is noise. Remember whether it ever did.
  const [offered, setOffered] = useState(shown < total);
  useEffect(() => {
    if (shown < total) setOffered(true);
  }, [shown, total]);

  if (total <= 0) return null;

  const remaining = Math.max(0, total - shown);
  const stepSize = Math.min(increment, remaining);

  const allLoaded = shown >= total;
  // An explicit terminalLabel is a message of its own, so it always shows.
  if (allLoaded && !offered && terminalLabel === undefined) return null;
  const buttonLabel = label ?? `Load ${stepSize} more`;
  const countText = countLabel ?? `Showing ${shown.toLocaleString()} of ${total.toLocaleString()}`;
  const terminalText = terminalLabel ?? `All ${total.toLocaleString()} loaded`;

  return (
    <div
      className={cn(
        "mt-10 flex flex-col items-center gap-3 md:mt-14",
        className,
      )}
    >
      {allLoaded ? (
        <Kicker tone="soft" tnum aria-live="polite">
          {terminalText}
        </Kicker>
      ) : (
        <>
          <Kicker tnum aria-live="polite">
            {countText}
            {countSuffix ? <span className="ml-1">{countSuffix}</span> : null}
          </Kicker>
          <button
            type="button"
            onClick={onLoadMore}
            disabled={isLoading}
            className={cn(
              "min-h-[44px] rounded-full border border-line bg-bone-deep px-6 text-ink",
              "font-sans text-sm font-medium tracking-tight",
              "transition-colors hover:bg-line/60",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
              "disabled:cursor-not-allowed disabled:opacity-50",
            )}
            aria-label={isLoading ? "Loading more" : buttonLabel}
          >
            {isLoading ? "Loading…" : buttonLabel}
          </button>
        </>
      )}
    </div>
  );
}
