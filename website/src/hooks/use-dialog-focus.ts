"use client";

import { useEffect, useRef, useSyncExternalStore } from "react";

const FOCUSABLE =
  'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

// Open dialogs, newest last: only the top one traps Tab.
const openDialogs: HTMLElement[] = [];
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((l) => l());
const subscribe = (l: () => void) => {
  listeners.add(l);
  return () => listeners.delete(l);
};

/** True while any Modal or drawer is open; toasts move off its footer (AX-53). */
export function useAnyDialogOpen(): boolean {
  return useSyncExternalStore(subscribe, () => openDialogs.length > 0, () => false);
}

function focusables(root: HTMLElement): HTMLElement[] {
  // Skip controls a disabled fieldset turns off and hidden ones (jsdom has no checkVisibility).
  return Array.from(root.querySelectorAll<HTMLElement>(FOCUSABLE)).filter(
    (el) => !el.matches(":disabled") && (typeof el.checkVisibility !== "function" || el.checkVisibility()),
  );
}

/**
 * Keyboard focus for a dialog (AX-1, audit X1). On open, focus moves inside
 * (unless a field already took it with autoFocus); with `trap`, Tab cycles
 * inside; on close, focus returns to the element that had it before.
 * Pass the element itself (kept in state through a callback ref), so the
 * effect runs again once a panel that mounts a render late appears.
 */
export function useDialogFocus(
  container: HTMLElement | null,
  open: boolean,
  { trap = false, initialFocus = "first" }: { trap?: boolean; initialFocus?: "first" | "container" } = {},
): void {
  // Recent focus targets: an autoFocus field inside takes focus before the
  // effect below runs, so the opener is the newest one outside the dialog.
  const recent = useRef<Element[]>([]);
  useEffect(() => {
    // A dialog mounted only while open (Modal shows its panel a render later): the opener has focus now.
    const now = document.activeElement;
    if (now && now !== document.body) recent.current = [now];
    const onFocusIn = (e: FocusEvent) => {
      if (e.target instanceof Element) recent.current = [e.target, ...recent.current.slice(0, 7)];
    };
    document.addEventListener("focusin", onFocusIn);
    return () => document.removeEventListener("focusin", onFocusIn);
  }, []);

  useEffect(() => {
    if (!open || !container) return;
    const active = document.activeElement;
    const opener =
      active && active !== document.body && !container.contains(active)
        ? active
        : (recent.current.find((el) => !container.contains(el)) ?? null);

    if (!container.contains(document.activeElement)) {
      const target = initialFocus === "container" ? container : (focusables(container)[0] ?? container);
      target.focus({ preventScroll: true });
    }

    openDialogs.push(container);
    notify();
    const onKeyDown = (e: KeyboardEvent) => {
      if (!trap || e.key !== "Tab" || openDialogs[openDialogs.length - 1] !== container) return;
      const items = focusables(container);
      const current = document.activeElement;
      const outside = !container.contains(current);
      if (items.length === 0) {
        e.preventDefault();
        container.focus({ preventScroll: true });
      } else if (e.shiftKey && (outside || current === items[0] || current === container)) {
        e.preventDefault();
        items[items.length - 1].focus();
      } else if (!e.shiftKey && (outside || current === items[items.length - 1])) {
        e.preventDefault();
        items[0].focus();
      }
    };
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      const at = openDialogs.indexOf(container);
      if (at >= 0) openDialogs.splice(at, 1);
      notify();
      // Leave focus alone if it already sits in another dialog that is still open.
      const now = document.activeElement;
      const free = !now || now === document.body || container.contains(now);
      if (free && opener instanceof HTMLElement && opener.isConnected) opener.focus({ preventScroll: true });
    };
  }, [container, open, trap, initialFocus]);
}
