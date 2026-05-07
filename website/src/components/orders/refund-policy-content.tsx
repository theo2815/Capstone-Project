import { Kicker } from "@/components/ui/kicker";
import { REFUND_POLICY_BULLETS } from "@/lib/refund-policy";

// Pure presentational. Used by RefundModal in both policy mode (full body)
// and request mode (collapsed <details> disclosure). One source of truth
// for the words.
export function RefundPolicyContent() {
  return (
    <ol className="space-y-5">
      {REFUND_POLICY_BULLETS.map((bullet, i) => (
        <li
          key={bullet.kicker}
          className="border-l border-line pl-4 md:pl-5"
        >
          <Kicker as="p" tnum>
            {String(i + 1).padStart(2, "0")} · {bullet.kicker}
          </Kicker>
          <p className="mt-2 font-sans text-sm text-ink-soft leading-relaxed">
            {bullet.body}
          </p>
        </li>
      ))}
    </ol>
  );
}
