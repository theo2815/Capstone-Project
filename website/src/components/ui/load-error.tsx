import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

/** A failed or offline read: says so and offers Retry, instead of a skeleton that never ends. */
export function LoadError({ what, paused, onRetry, className }: {
  /** Lower-case noun phrase, e.g. "your events". */
  what: string;
  paused?: boolean;
  onRetry: () => void;
  className?: string;
}) {
  return (
    <div role="alert" className={cn("font-sans text-base text-slate", className)}>
      <p>{paused ? `You're offline. Reconnect to load ${what}, or retry once connected.` : `Couldn't load ${what}.`}</p>
      <button type="button" onClick={onRetry} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-4")}>Retry</button>
    </div>
  );
}

/** True when a read has no data and won't get any without a retry. */
export const loadFailed = (q: { data?: unknown; isError: boolean; isPaused: boolean }) =>
  q.data == null && (q.isError || q.isPaused);
