"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useConfirmation } from "@/hooks/use-confirmation";
import type { ConfirmConfig } from "@/store/confirmation-store";

// While `active`, warn before work is lost: the browser's own prompt for
// reload/close, and a confirm for in-app links (client-side navigation never
// fires beforeunload). Browser back/forward is not caught.
export function useLeaveGuard(active: boolean, config: ConfirmConfig) {
  const { confirm } = useConfirmation();
  const router = useRouter();
  const { title, message, confirmLabel, cancelLabel } = config;

  useEffect(() => {
    if (!active) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    const onClick = (e: MouseEvent) => {
      if (e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = (e.target as Element | null)?.closest?.("a[href]") as HTMLAnchorElement | null;
      if (!a || a.target === "_blank" || a.origin !== window.location.origin) return;
      e.preventDefault();
      e.stopPropagation();
      void confirm({ title, message, confirmLabel, cancelLabel, danger: true })
        .then((ok) => { if (ok) router.push(a.pathname + a.search + a.hash); });
    };
    window.addEventListener("beforeunload", warn);
    document.addEventListener("click", onClick, true);
    return () => {
      window.removeEventListener("beforeunload", warn);
      document.removeEventListener("click", onClick, true);
    };
    // confirm is a fresh function each render; the store it opens is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active, title, message, confirmLabel, cancelLabel, router]);
}
