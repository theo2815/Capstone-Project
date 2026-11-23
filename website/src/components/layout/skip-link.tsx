"use client";

import type { MouseEvent } from "react";

// First focusable element on every page (runner audit P3-2): hidden until a
// keyboard user tabs to it, then it jumps focus past the header to the page's
// <main>. Pages don't share a main id, so it finds the element instead. A shell
// whose <main> also wraps the header and nav (admin) marks its content id="main".
export function SkipLink() {
  const skip = (e: MouseEvent<HTMLAnchorElement>) => {
    const main = document.getElementById("main") ?? document.querySelector("main");
    if (!main) return;
    e.preventDefault();
    if (!main.hasAttribute("tabindex")) main.setAttribute("tabindex", "-1");
    main.focus(); // focusing also scrolls it into view
  };
  return (
    <a
      href="#main"
      onClick={skip}
      className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-full focus:bg-ink focus:px-5 focus:py-3 focus:font-display focus:font-bold focus:text-[15px] focus:text-surface"
    >
      Skip to content
    </a>
  );
}
