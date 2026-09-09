"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { setStepUpHandler } from "@/lib/api";
import { StepUpModal } from "@/components/auth/step-up-modal";

// Registers the app-wide step-up handler with the api client and renders the
// prompt. When a request 403s with STEP_UP_REQUIRED, the client awaits this
// handler, which opens the modal and resolves with a fresh token (or null on
// cancel) so the client can retry with X-Step-Up. Inert until that happens.
export function StepUpProvider() {
  const [open, setOpen] = useState(false);
  const resolverRef = useRef<((token: string | null) => void) | null>(null);
  const pendingRef = useRef<Promise<string | null> | null>(null);

  useEffect(() => {
    setStepUpHandler(
      () => {
        if (pendingRef.current) return pendingRef.current;
        pendingRef.current = new Promise<string | null>((resolve) => {
          resolverRef.current = resolve;
          setOpen(true);
        });
        return pendingRef.current;
      },
    );
    return () => {
      setStepUpHandler(null);
      resolverRef.current?.(null);
      resolverRef.current = null;
      pendingRef.current = null;
    };
  }, []);

  const handleResolve = useCallback((token: string | null) => {
    setOpen(false);
    const resolve = resolverRef.current;
    resolverRef.current = null;
    pendingRef.current = null;
    resolve?.(token);
  }, []);

  return <StepUpModal open={open} onResolve={handleResolve} />;
}
