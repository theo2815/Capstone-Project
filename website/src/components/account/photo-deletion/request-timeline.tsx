import { formatLongDate } from "@/lib/format";
import { runnerTimelineLabel } from "@/lib/runner-removal-status";
import { cn } from "@/lib/utils";
import type { RunnerRemovalRequestDetailDto } from "@/types/photo-removal";

// RD-12: what happened to a runner's request, oldest first. Words and times only.
export function RequestTimeline({ steps }: { steps: RunnerRemovalRequestDetailDto["timeline"] }) {
  if (steps.length === 0) return null;
  return (
    <ol aria-label="What happened so far" className="border border-line rounded-xl bg-bone-deep/30 divide-y divide-line/60">
      {steps.map((s, i) => (
        <li key={`${s.step}-${s.at}`} className="flex flex-wrap items-center gap-x-3 gap-y-1 px-4 md:px-5 py-3 font-sans text-sm">
          <span aria-hidden="true" className={cn("size-1.5 rounded-full shrink-0", i === steps.length - 1 ? "bg-fresh" : "bg-slate-soft")} />
          <span className="text-ink">{runnerTimelineLabel(s.step)}</span>
          <time dateTime={s.at} className="ml-auto font-mono tnum text-slate">{formatStepTime(s.at)}</time>
        </li>
      ))}
    </ol>
  );
}

function formatStepTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${formatLongDate(iso)} · ${d.toLocaleTimeString("en-PH", { hour: "2-digit", minute: "2-digit", hour12: false })}`;
}
