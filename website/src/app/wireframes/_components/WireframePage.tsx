import Link from "next/link";
import { ReactNode } from "react";

export function WireframePage({
  module,
  ucId,
  title,
  tracesTo,
  mustShow,
  children,
}: {
  module: "M1" | "M2" | "M3";
  ucId: string;
  title: string;
  tracesTo: string;
  mustShow: string[];
  children: ReactNode;
}) {
  const moduleLabel: Record<string, string> = {
    M1: "Module 1 — Mobile Application",
    M2: "Module 2 — Desktop Application (BatchMyPhotos)",
    M3: "Module 3 — Marketplace & AI Search",
  };

  return (
    <div className="min-h-screen bg-[#FAFAF7] py-10 px-6 print:bg-white">
      <div className="mx-auto max-w-6xl">
        {/* header strip */}
        <div className="mb-6 flex items-center justify-between border-b border-neutral-900 pb-3">
          <Link
            href="/wireframes"
            className="font-mono text-[10px] uppercase tracking-wider text-neutral-500 hover:text-neutral-900"
          >
            ← Index
          </Link>
          <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
            QuickPitik · SRS Wireframe
          </div>
        </div>

        {/* title block — paper-style figure header */}
        <div className="mb-6 grid grid-cols-12 items-end gap-4 border-b border-neutral-300 pb-4">
          <div className="col-span-12 md:col-span-8">
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-500">
              {moduleLabel[module]}
            </div>
            <h1 className="mt-1 font-display text-2xl font-semibold text-neutral-900">
              <span className="text-neutral-400">{ucId}</span> · {title}
            </h1>
          </div>
          <div className="col-span-12 md:col-span-4 md:text-right">
            <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
              Traces to
            </div>
            <div className="font-mono text-xs text-neutral-800">{tracesTo}</div>
          </div>
        </div>

        {/* wireframe stage */}
        <div className="rounded-lg border border-dashed border-neutral-300 bg-white p-8">
          {children}
        </div>

        {/* must-show contract */}
        <div className="mt-6 grid grid-cols-12 gap-4 border-t border-neutral-300 pt-4">
          <div className="col-span-12 md:col-span-3">
            <div className="font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-500">
              Must show
            </div>
            <div className="font-mono text-[10px] text-neutral-400">
              From SRS placeholder
            </div>
          </div>
          <ul className="col-span-12 md:col-span-9 space-y-1">
            {mustShow.map((m, i) => (
              <li
                key={i}
                className="font-mono text-[11px] text-neutral-700 leading-relaxed"
              >
                <span className="mr-2 text-neutral-400">{String(i + 1).padStart(2, "0")}</span>
                {m}
              </li>
            ))}
          </ul>
        </div>

        <div className="mt-6 flex justify-between font-mono text-[10px] uppercase tracking-wider text-neutral-400">
          <span>Figure · {ucId}</span>
          <span>Low-fidelity wireframe · not final UI</span>
        </div>
      </div>
    </div>
  );
}
