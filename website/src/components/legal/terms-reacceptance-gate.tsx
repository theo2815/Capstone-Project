"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { LegalText } from "@/components/legal/legal-document";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { Modal } from "@/components/ui/modal";
import { useAuth } from "@/hooks/use-auth";
import { acceptTerms } from "@/lib/api-account";
import { ROUTES } from "@/lib/constants";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { LEGAL_VERSION } from "@/lib/legal/version";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/store/auth-store";

const noop = () => {};

// RV-L5 (plan 5.14, O3): a photographer the server marks termsAcceptanceRequired cannot use the
// dashboard until they accept the current Terms or sign out. No close button, Escape or backdrop.
export function TermsReacceptanceGate() {
  const user = useAuthStore((s) => s.user);
  const setUser = useAuthStore((s) => s.setUser);
  const { logout } = useAuth();
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const open = user?.termsAcceptanceRequired === true;

  // The shared Modal has no focus trap; keep Tab inside the prompt.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Tab" || !bodyRef.current) return;
      const items = Array.from(bodyRef.current.querySelectorAll<HTMLElement>("a[href], button:not([disabled])"));
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      const inside = bodyRef.current.contains(active);
      if (e.shiftKey ? active === first || !inside : active === last || !inside) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open]);

  async function accept() {
    setBusy(true);
    setError(null);
    try {
      setUser(await acceptTerms(LEGAL_VERSION));
    } catch (err) {
      // The server words every refusal (409 CONSENT_VERSION_MISMATCH asks for a reload).
      setError(err instanceof Error ? err.message : null);
    } finally {
      setBusy(false);
    }
  }

  function signOut() {
    logout();
    router.replace(ROUTES.LOGIN);
  }

  return (
    <Modal isOpen={open} onClose={noop} title={LEGAL_COPY.reacceptTitle} hideClose escDisabled>
      <div ref={bodyRef} className="space-y-5 font-sans text-sm leading-relaxed text-ink-soft">
        <p>{LEGAL_COPY.reacceptBody}</p>
        <p>
          <LegalText text={LEGAL_COPY.reacceptLinks} newTab />
        </p>
        <div className="space-y-3 border-t border-line pt-5">
          <FieldError message={error} id="reaccept-error" density="tight" />
          <button
            type="button"
            autoFocus
            onClick={accept}
            disabled={busy}
            aria-describedby={error ? "reaccept-error" : undefined}
            className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-full")}
          >
            {LEGAL_COPY.reacceptButton}
          </button>
          <p className="text-slate">{LEGAL_COPY.reacceptButtonLine}</p>
          <p className="text-slate">
            {LEGAL_COPY.reacceptSignOutLead}{" "}
            <button
              type="button"
              onClick={signOut}
              className="min-h-11 font-semibold text-ink underline decoration-line-strong underline-offset-4 transition-colors hover:decoration-fresh"
            >
              {LEGAL_COPY.reacceptSignOut}
            </button>
            .
          </p>
        </div>
      </div>
    </Modal>
  );
}
