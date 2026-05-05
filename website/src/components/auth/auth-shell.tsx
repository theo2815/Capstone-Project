import type { ReactNode } from "react";
import { SiteHeader } from "@/components/layout/site-header";

interface AuthShellProps {
  rightLink: { label: string; href: string };
  children: ReactNode;
}

export function AuthShell({ rightLink, children }: AuthShellProps) {
  return (
    <>
      <SiteHeader rightLink={rightLink} />
      <main className="flex-1 relative flex flex-col">
        <AmbientHorizon />
        <div className="relative z-10 flex-1 px-6 md:px-10 pt-12 md:pt-20 pb-16 flex flex-col items-center">
          <div className="w-full max-w-[440px]">{children}</div>
        </div>
        <div className="relative z-10 px-6 md:px-10 pb-10">
          <p className="text-center font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
            Cebu &middot; PH &middot; 2026
          </p>
        </div>
      </main>
    </>
  );
}

function AmbientHorizon() {
  return (
    <div
      aria-hidden="true"
      className="pointer-events-none absolute inset-x-0 bottom-24 h-px"
    >
      <svg
        viewBox="0 0 1000 1"
        preserveAspectRatio="none"
        className="w-full h-px"
        fill="none"
      >
        <path
          d="M 0 0.5 L 1000 0.5"
          stroke="var(--fresh)"
          strokeWidth="1"
          strokeOpacity="0.3"
          strokeLinecap="round"
          className="draw-line"
          style={{ "--dash": "1000" } as React.CSSProperties}
        />
      </svg>
    </div>
  );
}
