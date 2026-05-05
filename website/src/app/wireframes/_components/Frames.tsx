import { ReactNode } from "react";

/* ─────────────────────────────────────────────
   Device frames — phone (mobile), desktop window,
   browser. All shown as low-fi outlines so the
   inner UI reads as a wireframe, not a render.
   ───────────────────────────────────────────── */

export function PhoneFrame({
  children,
  statusLabel,
  className = "",
}: {
  children: ReactNode;
  statusLabel?: string;
  className?: string;
}) {
  return (
    <div className={["relative", className].join(" ")}>
      <div className="mx-auto w-[360px] rounded-[36px] border-2 border-neutral-900 bg-white p-2 shadow-[0_0_0_1px_rgba(0,0,0,0.05),0_30px_60px_-30px_rgba(0,0,0,0.25)]">
        {/* notch */}
        <div className="mx-auto mb-1 h-4 w-24 rounded-b-2xl bg-neutral-900" />
        <div className="overflow-hidden rounded-[24px] border border-neutral-300 bg-white">
          {/* status bar */}
          <div className="flex items-center justify-between bg-neutral-50 px-4 py-1.5 font-mono text-[10px] uppercase tracking-wider text-neutral-500">
            <span>9:41</span>
            <span>{statusLabel ?? "QUICKPITIK · ANDROID"}</span>
            <span>5G  ▮▮▮</span>
          </div>
          <div className="min-h-[640px] bg-white">{children}</div>
        </div>
        {/* home indicator */}
        <div className="mx-auto mt-2 h-1 w-24 rounded-full bg-neutral-900" />
      </div>
    </div>
  );
}

export function DesktopFrame({
  children,
  appLabel,
  className = "",
}: {
  children: ReactNode;
  appLabel?: string;
  className?: string;
}) {
  return (
    <div className={["mx-auto w-full max-w-5xl", className].join(" ")}>
      <div className="overflow-hidden rounded-md border-2 border-neutral-900 bg-white shadow-[0_30px_60px_-30px_rgba(0,0,0,0.25)]">
        {/* title bar */}
        <div className="flex items-center justify-between border-b border-neutral-900 bg-neutral-100 px-3 py-2">
          <div className="flex items-center gap-1.5">
            <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
            <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
            <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
          </div>
          <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-700">
            {appLabel ?? "BatchMyPhotos — Desktop (Electron)"}
          </div>
          <div className="font-mono text-[10px] text-neutral-400">v1.0</div>
        </div>
        <div className="min-h-[600px] bg-white">{children}</div>
      </div>
    </div>
  );
}

export function BrowserFrame({
  children,
  url,
  className = "",
}: {
  children: ReactNode;
  url?: string;
  className?: string;
}) {
  return (
    <div className={["mx-auto w-full max-w-6xl", className].join(" ")}>
      <div className="overflow-hidden rounded-md border-2 border-neutral-900 bg-white shadow-[0_30px_60px_-30px_rgba(0,0,0,0.25)]">
        {/* tab strip */}
        <div className="flex items-center gap-2 border-b border-neutral-900 bg-neutral-100 px-3 py-2">
          <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
          <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
          <span className="h-3 w-3 rounded-full border border-neutral-900 bg-white" />
          <div className="ml-3 flex h-6 max-w-md flex-1 items-center rounded-md border border-neutral-300 bg-white px-2 font-mono text-[10px] text-neutral-600">
            <span className="mr-2 text-neutral-400">https://</span>
            <span className="truncate">{url ?? "quickpitik.ph"}</span>
          </div>
        </div>
        <div className="min-h-[640px] bg-white">{children}</div>
      </div>
    </div>
  );
}
