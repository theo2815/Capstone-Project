"use client";

import { useEffect, useState } from "react";

// IntersectionObserver-driven "which section is on screen" hook. Used by the
// profile-shell IdentityRail jump-to nav so the active dot tracks scroll.
// `ids` MUST be stable (e.g., a top-level const) — passing a fresh array each
// render re-subscribes every paint.
export function useActiveSection(
  ids: ReadonlyArray<string>,
): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        rootMargin: "-25% 0px -55% 0px",
        threshold: [0, 0.1, 0.5, 1],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [ids]);
  return active;
}
