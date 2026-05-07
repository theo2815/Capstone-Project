import { cn } from "@/lib/utils";
import { Kicker } from "./kicker";

interface CharCounterProps {
  current: number;
  max: number;
  /** Stable id so a sibling input can chain via aria-describedby. Counter
   *  itself is decorative for SR (errors are the announcement channel). */
  id?: string;
  className?: string;
}

// Right-aligned mono current/max display for length-bounded inputs. Reuses
// <Kicker> for the mono + uppercase + tnum baseline (numerals only — uppercase
// is a no-op on digits and the slash).
//
// Color thresholds (locked in Slice 1A, inherited by every length-bounded
// input that ships after):
//   < 80%     → text-slate-soft (default)
//   ≥ 80%    → text-warning (the amber/orange #D97706)
//   ≥ 100%   → text-error (red #DC2626; user has hit cap, next keystroke
//             blocked or surfaces an inline FieldError)
//
// Decorative for screen readers — pair with <FieldError> for any actual
// constraint violation. Counter is the at-a-glance signal; error is the
// announcement.
export function CharCounter({
  current,
  max,
  id,
  className,
}: CharCounterProps) {
  const ratio = max > 0 ? current / max : 0;
  const tone =
    ratio >= 1
      ? "text-error"
      : ratio >= 0.8
        ? "text-warning"
        : undefined;

  return (
    <Kicker
      as="p"
      id={id}
      tone="soft"
      tnum
      aria-hidden="true"
      className={cn("text-right", tone, className)}
    >
      {current}/{max}
    </Kicker>
  );
}
