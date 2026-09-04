"use client";

import { type ReactNode } from "react";
import { cn } from "@/lib/utils";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";

type TooltipPosition =
  | "top"
  | "bottom"
  | "left"
  | "right"
  | "bottom-end"
  | "top-end";

interface TooltipProps {
  label: string;
  position?: TooltipPosition;
  children: ReactNode;
  className?: string;
  bubbleClassName?: string;
}

// `*-end` variants anchor the bubble to the wrapper's right edge instead of
// centering it. Use these when the trigger sits at a container edge
// (e.g., `absolute right-4`) so the bubble extends inward and won't be
// clipped by overflow-hidden parents.
const POSITION_CLASS: Record<TooltipPosition, string> = {
  top: "bottom-full left-1/2 -translate-x-1/2 -translate-y-2",
  bottom: "top-full left-1/2 -translate-x-1/2 translate-y-2",
  left: "right-full top-1/2 -translate-y-1/2 -translate-x-2",
  right: "left-full top-1/2 -translate-y-1/2 translate-x-2",
  "bottom-end": "top-full right-0 translate-y-2",
  "top-end": "bottom-full right-0 -translate-y-2",
};

// CSS-only tooltip — no portals, no JS positioning. Uses bone surface so it
// reads cleanly on both light and dark backgrounds (e.g., the dark image area
// behind the SaveButton on event cards). Trigger should carry its own
// aria-label; the tooltip is purely visual.
//
// Named group `group/tooltip` is critical: callers often live inside a card
// with `<Link className="... group ...">` (so the "Open →" text can deepen
// on card hover). Without a named group, Tailwind's `group-hover:` would
// match ANY ancestor `.group`, and hovering the card would pop the tooltip
// even when the cursor is nowhere near the trigger. Namespacing the group
// scopes the listener to this Tooltip's own wrapper only.
export function Tooltip({
  label,
  position = "top",
  children,
  className,
  bubbleClassName,
}: TooltipProps) {
  return (
    <span className={cn("relative inline-flex group/tooltip", className)}>
      {children}
      <span
        aria-hidden="true"
        className={cn(
          "absolute pointer-events-none z-30 whitespace-nowrap",
          "bg-bone border border-line text-ink",
          "font-mono uppercase py-1.5 px-2.5 rounded-full",
          KICKER_SIZE_CLASS.sm,
          "shadow-[0_4px_12px_-4px_rgba(17,17,17,0.18)]",
          "opacity-0 scale-95 transition-[opacity,transform] duration-150",
          "group-hover/tooltip:opacity-100 group-hover/tooltip:scale-100",
          "group-focus-within/tooltip:opacity-100 group-focus-within/tooltip:scale-100",
          POSITION_CLASS[position],
          bubbleClassName,
        )}
      >
        {label}
      </span>
    </span>
  );
}
