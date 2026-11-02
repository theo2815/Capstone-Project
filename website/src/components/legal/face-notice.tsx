"use client";

import { LegalText } from "@/components/legal/legal-document";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { cn } from "@/lib/utils";
import { useFaceConsent, useFaceConsentStore } from "@/store/face-consent-store";

// Shown above every selfie search or save until ticked once on this device.
// The link opens in a new tab so a half-done search is not lost.
export function FaceNotice({ className }: { className?: string }) {
  const accepted = useFaceConsent();
  const accept = useFaceConsentStore((s) => s.accept);
  if (accepted) return null;
  return (
    <div className={cn("rounded-xl border border-line bg-bone-deep px-5 py-4 font-sans text-sm leading-relaxed text-ink-soft", className)}>
      <p>
        <LegalText text={LEGAL_COPY.faceNotice} newTab />
      </p>
      <label className="mt-3 flex min-h-11 cursor-pointer items-center gap-3 text-ink select-none">
        <input
          type="checkbox"
          checked={false}
          onChange={(e) => e.target.checked && accept()}
          className="size-4 shrink-0 accent-ink"
        />
        <span>{LEGAL_COPY.faceTickLabel}</span>
      </label>
    </div>
  );
}
