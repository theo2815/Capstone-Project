"use client";

import { LegalText } from "@/components/legal/legal-document";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { cn } from "@/lib/utils";
import { useBibConsent, useBibConsentStore, useFaceConsent, useFaceConsentStore } from "@/store/face-consent-store";

// Shown above every selfie search or save until ticked once on this device.
export function FaceNotice({ className }: { className?: string }) {
  const accept = useFaceConsentStore((s) => s.accept);
  if (useFaceConsent()) return null;
  return <Notice text={LEGAL_COPY.faceNotice} label={LEGAL_COPY.faceTickLabel} onAccept={accept} className={className} />;
}

// The same, above every bib search (task BT-1).
export function BibNotice({ className }: { className?: string }) {
  const accept = useBibConsentStore((s) => s.accept);
  if (useBibConsent()) return null;
  return <Notice text={LEGAL_COPY.bibNotice} label={LEGAL_COPY.bibTickLabel} onAccept={accept} className={className} />;
}

// The link opens in a new tab so a half-done search is not lost.
function Notice({ text, label, onAccept, className }: { text: string; label: string; onAccept: () => void; className?: string }) {
  return (
    <div className={cn("rounded-xl border border-line bg-bone-deep px-5 py-4 font-sans text-sm leading-relaxed text-ink-soft", className)}>
      <p>
        <LegalText text={text} newTab />
      </p>
      <label className="mt-3 flex min-h-11 cursor-pointer items-center gap-3 text-ink select-none">
        <input
          type="checkbox"
          checked={false}
          onChange={(e) => e.target.checked && onAccept()}
          className="size-4 shrink-0 accent-ink"
        />
        <span>{label}</span>
      </label>
    </div>
  );
}
