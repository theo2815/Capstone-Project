import type { MouseEvent } from "react";

// Cheap deterrence for preview <img>s: no drag-to-desktop, no right-click
// "Save image as…", no iOS long-press callout, no text-selection ghost. The
// server-baked watermark is the real protection — this just removes the
// one-gesture paths. Spread PROTECTED_IMG_PROPS onto the element and add
// PROTECTED_IMG_CLASS to its className.
export const PROTECTED_IMG_PROPS = {
  draggable: false,
  onContextMenu: (e: MouseEvent<HTMLImageElement>) => e.preventDefault(),
} as const;

export const PROTECTED_IMG_CLASS = "select-none [-webkit-touch-callout:none]";
