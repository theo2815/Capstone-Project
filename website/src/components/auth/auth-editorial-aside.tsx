export function AuthEditorialAside() {
  return (
    <aside
      className="hidden md:flex flex-col justify-between min-h-[28rem] pl-2 lg:pl-8"
      aria-hidden="true"
    >
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik / 2026
      </p>

      <div className="space-y-6">
        <p className="font-display text-3xl lg:text-4xl font-medium tracking-tight leading-[1.15] text-ink-soft max-w-md">
          Race photos,
          <br />
          <span className="text-ink">delivered in minutes.</span>
        </p>
        <svg
          viewBox="0 0 280 8"
          className="w-40 h-2"
          fill="none"
          aria-hidden="true"
        >
          <path
            d="M 0 4 L 280 4"
            stroke="var(--fresh)"
            strokeWidth="1.5"
            strokeLinecap="round"
            className="draw-line"
            style={{ "--dash": "280" } as React.CSSProperties}
          />
        </svg>
        <p className="font-sans text-sm text-slate max-w-sm">
          Cebu&apos;s marathon photography platform. Find your photos, or sell yours.
        </p>
      </div>

      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        Cebu &middot; PH &middot; 2026
      </p>
    </aside>
  );
}
