"use client";

import { type MouseEvent } from "react";
import { useRouter, usePathname } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useToast } from "@/hooks/use-toast";
import { getEventById } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { Tooltip } from "@/components/ui/tooltip";

type SaveButtonVariant = "card" | "inline";

interface SaveButtonProps {
  eventId: string;
  variant?: SaveButtonVariant;
  className?: string;
}

export function SaveButton({
  eventId,
  variant = "card",
  className,
}: SaveButtonProps) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const isSaved = useSavedEventsStore((s) => s.ids.includes(eventId));
  const toggle = useSavedEventsStore((s) => s.toggle);
  const { showToast } = useToast();
  const router = useRouter();
  const pathname = usePathname();

  function handleClick(e: MouseEvent<HTMLButtonElement>) {
    // Prevent the wrapping <Link> on event cards from firing.
    e.preventDefault();
    e.stopPropagation();

    if (!isAuthenticated) {
      router.push(
        `${ROUTES.LOGIN}?redirect=${encodeURIComponent(pathname || ROUTES.EVENTS)}`,
      );
      return;
    }

    // TODO(backend): swap for `api.post("/me/saved-events", { eventId })` /
    // `api.delete("/me/saved-events/${eventId}")` when Spring Boot Phase B lands.
    const willBeSaved = !isSaved;
    toggle(eventId);

    const eventName = getEventById(eventId)?.name ?? "this event";
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
        onClick: () => toggle(eventId),
      },
    });
  }

  if (variant === "inline") {
    return (
      <button
        type="button"
        onClick={handleClick}
        aria-pressed={isSaved}
        aria-label={isSaved ? "Unsave this event" : "Save this event"}
        className={cn(
          "group inline-flex items-center gap-2 font-mono uppercase tracking-[0.3em] text-[10px] transition-colors rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone",
          isSaved
            ? "text-fresh hover:text-fresh-deep"
            : "text-slate hover:text-ink",
          className,
        )}
      >
        <BookmarkGlyph filled={isSaved} className="size-3.5" />
        <span>{isSaved ? "Saved" : "Save event"}</span>
      </button>
    );
  }

  return (
    <Tooltip
      label={isSaved ? "Saved · click to unsave" : "Save event"}
      position="bottom-end"
      className={cn("absolute top-4 right-4 z-10", className)}
    >
      <button
        type="button"
        onClick={handleClick}
        aria-pressed={isSaved}
        aria-label={isSaved ? "Unsave this event" : "Save this event"}
        className={cn(
          "size-9 rounded-full inline-flex items-center justify-center transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone shadow-[0_4px_12px_-4px_rgba(17,17,17,0.18)]",
          isSaved
            ? "bg-fresh text-bone hover:bg-fresh-deep"
            : "bg-bone/90 backdrop-blur text-ink hover:bg-bone",
        )}
      >
        <BookmarkGlyph filled={isSaved} className="size-4" />
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
