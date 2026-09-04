"use client";

import Script from "next/script";
import { useCallback, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api";
import { GOOGLE_CLIENT_ID, ROUTES } from "@/lib/constants";
import { roleHome } from "@/lib/redirect";
import { cn } from "@/lib/utils";

// Real Google sign-in via Google Identity Services (GIS). GIS renders its own
// branded button inside an iframe — there is no supported way to trigger the
// credential flow from a custom-styled button, so the 2026-05-05 hand-drawn
// one is retired by necessity (accepted tradeoff in the 2026-08-29 OAuth
// plan). The old boolean flag is env-derived now: leave
// NEXT_PUBLIC_GOOGLE_CLIENT_ID unset and this block disappears from /login +
// /register, mirroring the backend's blank-GOOGLE_CLIENT_ID 503.
export const OAUTH_ENABLED = GOOGLE_CLIENT_ID !== "";

const GSI_SRC = "https://accounts.google.com/gsi/client";
// GIS hard-caps its rendered button at 400 CSS px; the 440px auth card gets
// it slightly inset, centered by the flex wrapper below.
const GSI_MAX_WIDTH = 400;

interface GoogleButtonProps {
  disabled?: boolean;
}

export function GoogleButton({ disabled }: GoogleButtonProps) {
  const router = useRouter();
  const { googleLogin } = useAuth();
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  const handleCredential = useCallback(
    async (response: GsiCredentialResponse) => {
      setError(null);
      try {
        const user = await googleLogin(response.credential);
        if (user) {
          router.replace(roleHome(user.role));
        } else {
          // null = backend answered ROLE_REQUIRED: a brand-new Google account
          // that picks RUNNER/PHOTOGRAPHER on /onboarding first.
          router.replace(ROUTES.ONBOARDING);
        }
      } catch (err) {
        setError(
          err instanceof ApiError
            ? err.message
            : "Google sign-in failed. Try again.",
        );
      }
    },
    [googleLogin, router],
  );

  const init = useCallback(() => {
    const google = window.google;
    const container = containerRef.current;
    if (!google || !container) return;
    // onReady re-fires on every remount (and StrictMode double-invokes it in
    // dev); renderButton appends a fresh iframe per call, so clear first.
    container.innerHTML = "";
    google.accounts.id.initialize({
      client_id: GOOGLE_CLIENT_ID,
      callback: handleCredential,
    });
    google.accounts.id.renderButton(container, {
      type: "standard",
      theme: "outline",
      text: "continue_with",
      shape: "pill",
      logo_alignment: "center",
      width: Math.min(container.offsetWidth || GSI_MAX_WIDTH, GSI_MAX_WIDTH),
    });
  }, [handleCredential]);

  if (!OAUTH_ENABLED) return null;

  return (
    <div className="space-y-3">
      <Script src={GSI_SRC} strategy="afterInteractive" onReady={init} />
      {/* min-h reserves the GIS button's height so the card doesn't jump
          when the iframe lands. If the script never loads (ad blocker), the
          slot stays empty and email/password remains the path — deliberate. */}
      <div
        ref={containerRef}
        aria-disabled={disabled}
        className={cn(
          "flex justify-center min-h-[44px]",
          disabled && "pointer-events-none opacity-50",
        )}
      />
      {error && (
        <p role="alert" className="font-sans text-sm text-error text-center">
          {error}
        </p>
      )}
    </div>
  );
}

export function AuthDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3" aria-hidden="true">
      <span className="flex-1 border-t border-line" />
      <span className="font-mono uppercase tracking-[0.14em] text-[10px] text-slate-soft">
        {label}
      </span>
      <span className="flex-1 border-t border-line" />
    </div>
  );
}
