"use client";

import { type MouseEvent } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useToast } from "@/hooks/use-toast";
import { useEffectiveRole } from "@/hooks/use-effective-role";
import { getEventById } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { Tooltip } from "@/components/ui/tooltip";
import type { SavedEventSummary } from "@/lib/api-saved-events";

type SaveButtonVariant = "card" | "inline";

// Optimistic summary the SaveButton needs to push into the store on click so
// /profile race log gets a row immediately — before the POST round-trip. Same
// shape as SavedEventSummary minus savedAt (the store stamps that). All
// SaveButton call sites have these fields on the event object they already
// render, so threading is cheap.
export type SaveButtonEvent = {
  id: string;
  slug: string;
  name: string;
  date: string;
  state: SavedEventSummary["state"];
  bannerUrl?: string | null;
  location: string;
};

interface SaveButtonProps {
  eventId: string;
  /** Optional event summary used for the optimistic store insert. When
   *  omitted the store still toggles by ID, but the race-log row won't
   *  render until the BE POST response arrives. */
  event?: SaveButtonEvent;
  variant?: SaveButtonVariant;
  className?: string;
  /** When true, the button keeps its default unsaved visual but does not
   *  toggle or navigate. Tooltip swaps to "Runners only". Used on the public
   *  /{handle} page in the owner-preview branch — saving your own event is
   *  not a meaningful action for a photographer. */
  disabled?: boolean;
}

export function SaveButton({
  eventId,
  event,
  variant = "card",
  className,
  disabled = false,
}: SaveButtonProps) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  // Effective role so a photographer in runner view gets the save affordance
  // like a runner. (Persisted sync stays RUNNER-gated — saves are local-only
  // for a photographer account, but they still toggle + survive refresh.)
  const userRole = useEffectiveRole();
  const isSaved = useSavedEventsStore((s) => s.ids.includes(eventId));
  const toggle = useSavedEventsStore((s) => s.toggle);
  const { showToast } = useToast();
  const router = useRouter();
  const pathname = usePathname();

  // Photographers don't save events — that's a runner-side action. Default to
  // hiding the button entirely whenever a photographer is logged in. The one
  // exception is `disabled={true}` (set by /{handle} owner-preview), where
  // we keep the icon visible but non-interactive with a "Runners only"
  // tooltip — public visitors to the same URL still get the live save.
  if (userRole === "PHOTOGRAPHER" && !disabled) {
    return null;
  }

  function handleClick(e: MouseEvent<HTMLButtonElement>) {
    // Prevent the wrapping <Link> on event cards from firing.
    e.preventDefault();
    e.stopPropagation();

    if (disabled) return;

    if (!isAuthenticated) {
      router.push(
        `${ROUTES.LOGIN}?redirect=${encodeURIComponent(pathname || ROUTES.EVENTS)}`,
      );
      return;
    }

    const willBeSaved = !isSaved;
    const optimistic = event && {
      id: event.id,
      slug: event.slug,
      name: event.name,
      date: event.date,
      state: event.state,
      bannerUrl: event.bannerUrl ?? null,
      location: event.location,
      savedAt: new Date().toISOString(),
    };
    toggle(eventId, optimistic);

    const eventName = event?.name ?? getEventById(eventId)?.name ?? "this event";
    showToast({
      kind: "success",
      message: willBeSaved
        ? `Saved ${eventName}.`
        : `Removed ${eventName} from saved.`,
      // Link to race log only on save — on unsave, navigating there to verify
      // the row is gone is more confusing than useful (purchased events remain).
      link: willBeSaved
        ? { label: "View", href: `${ROUTES.PROFILE}#race-log` }
        : undefined,
      action: {
        label: "Undo",
        onClick: () => toggle(eventId, optimistic),
      },
    });
  }

  if (variant === "inline") {
    return (
      <button
        type="button"
        onClick={handleClick}
        aria-pressed={disabled ? undefined : isSaved}
        aria-disabled={disabled || undefined}
        aria-label={
          disabled
            ? "Saving events is for runners"
            : isSaved
              ? "Unsave this event"
              : "Save this event"
        }
        className={cn(
          "group inline-flex items-center gap-2 font-mono uppercase tracking-[0.14em] text-[10px] transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
          disabled
            ? "text-slate opacity-50 cursor-not-allowed"
            : isSaved
              ? "text-fresh hover:text-fresh-deep"
              : "text-slate hover:text-ink",
          className,
        )}
      >
        <BookmarkGlyph filled={!disabled && isSaved} className="size-3.5" />
        <span>{!disabled && isSaved ? "Saved" : "Save event"}</span>
      </button>
    );
  }

  // Card variant. When `disabled`, the bookmark dims to half opacity with a
  // not-allowed cursor so touch users see the no-op too (the tooltip only
  // reaches hover); the tooltip still explains the action belongs to runners.
  // The button stays focusable for keyboard users; aria-disabled signals the
  // no-op to AT.
  const tooltipLabel = disabled
    ? "Runners only"
    : isSaved
      ? "Saved · click to unsave"
      : "Save event";

  return (
    <Tooltip
      label={tooltipLabel}
      position="bottom-end"
      className={cn("absolute top-4 right-4 z-10", className)}
    >
      <button
        type="button"
        onClick={handleClick}
        aria-pressed={disabled ? undefined : isSaved}
        aria-disabled={disabled || undefined}
        aria-label={
          disabled
            ? "Saving events is for runners"
            : isSaved
              ? "Unsave this event"
              : "Save this event"
        }
        className={cn(
          "size-9 rounded-full inline-flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone shadow-[0_4px_12px_-4px_rgba(17,17,17,0.18)]",
          disabled
            ? "bg-bone/90 backdrop-blur text-ink opacity-50 cursor-not-allowed"
            : isSaved
              ? "bg-fresh text-surface hover:bg-fresh-deep"
              : "bg-bone/90 backdrop-blur text-ink hover:bg-bone",
        )}
      >
        <BookmarkGlyph filled={!disabled && isSaved} className="size-4" />
      </button>
    </Tooltip>
  );
}

function BookmarkGlyph({
  filled,
  className,
}: {
  filled: boolean;
  className?: string;
}) {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 16 16"
      fill={filled ? "currentColor" : "none"}
      className={className}
    >
      <path
        d="M3.5 2.5 V13.5 L8 10.5 L12.5 13.5 V2.5 A1 1 0 0 0 11.5 1.5 H4.5 A1 1 0 0 0 3.5 2.5 Z"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinejoin="round"
      />
    </svg>
  );
}
