import { FieldError } from "@/components/ui/field-error";
import { LegalText } from "@/components/legal/legal-document";
import { LEGAL_COPY } from "@/lib/legal/copy";

// The sign-up agreement box (register and Google onboarding). Links open in a
// new tab so the half-filled form, and onboarding's in-memory Google token, survive.
export function TermsTick({
  checked,
  onChange,
  error,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  error?: string | null;
}) {
  return (
    <div>
      <label className="flex items-start gap-3 font-sans text-sm text-ink-soft leading-relaxed">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          aria-invalid={!!error}
          aria-describedby={error ? "terms-error" : undefined}
          className="mt-1 size-4 shrink-0 accent-ink"
        />
        <span>
          <LegalText text={LEGAL_COPY.signupTickLabel} newTab />
        </span>
      </label>
      <FieldError message={error} id="terms-error" density="tight" />
    </div>
  );
}
