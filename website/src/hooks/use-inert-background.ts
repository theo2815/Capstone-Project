"use client";

import { useEffect, type RefObject } from "react";

// While an overlay portaled to <body> is open, every other top-level element
// becomes `inert`, so Tab and screen readers cannot reach the page behind it
// (runner audit P3-7: the cart and checkout drawers). Overlays can stack (the
// cart hands off to checkout), so the newest open one stays live and the rest
// go inert. Only the `inert` this hook set is ever removed; anything another
// component made inert, and the toast region, are left alone.
const MARK = "data-inert-background";
const open: HTMLElement[] = [];

function sync() {
  document.querySelectorAll(`[${MARK}]`).forEach((el) => {
    el.removeAttribute("inert");
    el.removeAttribute(MARK);
  });
  const top = open[open.length - 1];
  if (!top) return;
  for (const el of Array.from(document.body.children)) {
    if (el === top || el.contains(top) || el.hasAttribute("inert")) continue;
    if (el.matches('[role="region"][aria-label="Notifications"]')) continue;
    el.setAttribute("inert", "");
    el.setAttribute(MARK, "");
  }
}

export function useInertBackground(ref: RefObject<HTMLElement | null>, active: boolean) {
  useEffect(() => {
    const el = ref.current;
    if (!active || !el) return;
    open.push(el);
    sync();
    return () => {
      open.splice(open.indexOf(el), 1);
      sync();
    };
  }, [ref, active]);
}
