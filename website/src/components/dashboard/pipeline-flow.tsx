import { Fragment } from "react";
import { cn } from "@/lib/utils";

export interface PipelineNode {
  /** Mono kicker label — uppercased + tracked in render. */
  kicker: string;
  /** Big tnum number under the kicker. */
  count: number;
  /** Optional sub-line under the count. */
  caption?: string;
  /** Active node carries a small breathing fresh dot prefix on the kicker. */
  active?: boolean;
}

interface PipelineFlowProps {
  nodes: ReadonlyArray<PipelineNode>;
}

// Pipeline: any sequence of nodes (e.g. Uploaded → Live for the photographer
// dashboard). Adapts the /photographers BatchScreen [col_arrow_col] motif to
// a row that flips to a vertical chain on mobile. The active stage is the
// only fresh element (small breathing dot, excluded from the
// one-fresh-per-viewport rule).
export function PipelineFlow({ nodes }: PipelineFlowProps) {
  return (
    <div className="flex flex-col md:flex-row md:items-stretch gap-3 md:gap-3">
      {nodes.map((node, i) => (
        <Fragment key={node.kicker}>
          <Node node={node} />
          {i < nodes.length - 1 && <Connector />}
        </Fragment>
      ))}
    </div>
  );
}

function Node({ node }: { node: PipelineNode }) {
  return (
    <div
      className={cn(
        "flex-1 min-w-0 rounded-2xl border p-5 md:p-6 transition-colors",
        node.active
          ? "border-line bg-bone-deep/70"
          : "border-line bg-bone-deep/30",
      )}
    >
      <div className="flex items-center gap-2">
        {node.active && (
          <span
            aria-hidden="true"
            className="size-1.5 rounded-full bg-fresh breathe shrink-0"
          />
        )}
        <p
          className={cn(
            "font-mono uppercase tracking-[0.3em] text-[10px]",
            node.active ? "text-ink" : "text-slate",
          )}
        >
          {node.kicker}
        </p>
      </div>
      <p className="font-display text-4xl md:text-5xl font-medium tracking-tight tnum text-ink mt-3 leading-none">
        {node.count.toLocaleString()}
      </p>
      {node.caption && (
        <p className="font-sans text-xs md:text-sm text-slate mt-3 leading-snug">
          {node.caption}
        </p>
      )}
    </div>
  );
}

function Connector() {
  return (
    <div
      aria-hidden="true"
      className="flex items-center justify-center text-slate-soft md:px-1"
    >
      <span className="md:hidden text-base">↓</span>
      <span className="hidden md:inline text-lg">→</span>
    </div>
  );
}
