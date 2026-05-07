import { cn } from "@/lib/utils";

type FieldErrorDensity = "default" | "tight";

interface FieldErrorProps {
  message?: string | null;
  id: string;
  density?: FieldErrorDensity;
  className?: string;
}

// Inline validation message for form fields. Renders nothing when message is
// falsy — no reserved space, no layout shift on appearance.
//
// Tone convention (Quiet Studio): sentence case, end with period, name the
// constraint not the user error. Good: "Image must be under 8 MB." Bad: "You
// uploaded a file that's too large."
//
// Pair with the input via aria-describedby={id} and aria-invalid={!!message}
// on the input itself — the wiring is the consumer's job so screen readers can
// chain multiple descriptors (counter + error).
//
// aria-live="polite" (not assertive) — these are user-driven validation
// messages, not interrupts; assertive would talk over typing.
const DENSITY_CLASS: Record<FieldErrorDensity, string> = {
  default: "mt-3",
  tight: "mt-1.5",
};

export function FieldError({
  message,
  id,
  density = "default",
  className,
}: FieldErrorProps) {
  if (!message) return null;
  return (
    <p
      id={id}
      role="alert"
      aria-live="polite"
      className={cn(
        "font-sans text-sm text-error leading-snug",
        DENSITY_CLASS[density],
        className,
      )}
    >
      {message}
    </p>
  );
}
