import { ReactNode } from "react";

/* ─────────────────────────────────────────────
   Wireframe primitives — technical-blueprint
   look intended for SRS paper figures.
   ───────────────────────────────────────────── */

export function Box({
  children,
  className = "",
  dashed = false,
  filled = false,
}: {
  children?: ReactNode;
  className?: string;
  dashed?: boolean;
  filled?: boolean;
}) {
  return (
    <div
      className={[
        "border border-neutral-800",
        dashed ? "border-dashed" : "",
        filled ? "bg-neutral-100" : "bg-white",
        "p-3",
        className,
      ].join(" ")}
    >
      {children}
    </div>
  );
}

export function Placeholder({
  label,
  className = "",
  height = "h-32",
  diagonals = true,
}: {
  label: string;
  className?: string;
  height?: string;
  diagonals?: boolean;
}) {
  return (
    <div
      className={[
        "border border-dashed border-neutral-500 bg-neutral-50 relative flex items-center justify-center",
        height,
        className,
      ].join(" ")}
      style={
        diagonals
          ? {
              backgroundImage:
                "repeating-linear-gradient(135deg, rgba(120,120,120,0.06) 0 1px, transparent 1px 8px)",
            }
          : undefined
      }
    >
      <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
        {label}
      </span>
    </div>
  );
}

export function Btn({
  children,
  primary = false,
  className = "",
  small = false,
}: {
  children: ReactNode;
  primary?: boolean;
  className?: string;
  small?: boolean;
}) {
  return (
    <span
      className={[
        "inline-flex items-center justify-center font-medium border",
        small ? "px-3 py-1 text-[11px]" : "px-4 py-2 text-xs",
        primary
          ? "bg-neutral-900 text-white border-neutral-900"
          : "bg-white text-neutral-900 border-neutral-800",
        className,
      ].join(" ")}
    >
      {children}
    </span>
  );
}

export function Field({
  label,
  value = "",
  placeholder,
  className = "",
}: {
  label?: string;
  value?: string;
  placeholder?: string;
  className?: string;
}) {
  return (
    <label className={["block", className].join(" ")}>
      {label && (
        <span className="block font-mono text-[10px] uppercase tracking-wider text-neutral-500 mb-1">
          {label}
        </span>
      )}
      <span className="block w-full border-b border-neutral-700 bg-transparent px-1 py-1.5 text-sm text-neutral-800">
        {value || (
          <span className="text-neutral-400">{placeholder ?? "_"}</span>
        )}
      </span>
    </label>
  );
}

export function Tag({
  children,
  tone = "neutral",
}: {
  children: ReactNode;
  tone?: "neutral" | "ok" | "warn" | "err" | "info";
}) {
  const tones: Record<string, string> = {
    neutral: "bg-neutral-100 text-neutral-700 border-neutral-300",
    ok: "bg-emerald-50 text-emerald-800 border-emerald-300",
    warn: "bg-amber-50 text-amber-800 border-amber-300",
    err: "bg-rose-50 text-rose-800 border-rose-300",
    info: "bg-sky-50 text-sky-800 border-sky-300",
  };
  return (
    <span
      className={[
        "inline-flex items-center border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider",
        tones[tone],
      ].join(" ")}
    >
      {children}
    </span>
  );
}

export function Annot({
  n,
  children,
  className = "",
}: {
  n: number | string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={["flex gap-2 items-start", className].join(" ")}>
      <span className="shrink-0 mt-0.5 inline-flex h-5 w-5 items-center justify-center rounded-full border border-neutral-800 bg-white font-mono text-[10px] font-semibold text-neutral-900">
        {n}
      </span>
      <span className="font-mono text-[11px] leading-relaxed text-neutral-700">
        {children}
      </span>
    </div>
  );
}

export function Marker({ n }: { n: number | string }) {
  return (
    <span className="inline-flex h-5 w-5 items-center justify-center rounded-full border border-neutral-800 bg-white font-mono text-[10px] font-semibold text-neutral-900 align-middle">
      {n}
    </span>
  );
}

export function Divider({ label }: { label?: string }) {
  return (
    <div className="my-3 flex items-center gap-2">
      <div className="h-px flex-1 bg-neutral-300" />
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
          {label}
        </span>
      )}
      <div className="h-px flex-1 bg-neutral-300" />
    </div>
  );
}

export function Caption({ children }: { children: ReactNode }) {
  return (
    <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
      {children}
    </div>
  );
}

export function Title({ children }: { children: ReactNode }) {
  return (
    <div className="font-semibold text-sm text-neutral-900">{children}</div>
  );
}

export function Sub({ children }: { children: ReactNode }) {
  return <div className="text-[11px] text-neutral-600">{children}</div>;
}
