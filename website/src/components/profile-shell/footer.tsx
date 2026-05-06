// Mono kicker pair shared across profile-shell pages. Mounted at the bottom
// of /profile, /account, /orders, /dashboard.
export function ProfileShellFooter() {
  return (
    <footer className="px-6 md:px-10 py-8 mt-12 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}
