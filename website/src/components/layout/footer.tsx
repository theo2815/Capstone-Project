import Link from "next/link";
import { ROUTES } from "@/lib/constants";

export function Footer() {
  return (
    <footer className="border-t border-line bg-paper-deep">
      {/* Finish-line accent */}
      <div className="h-1 w-full flex" aria-hidden="true">
        <span className="flex-1 bg-fresh" />
        <span className="flex-1 bg-fresh-deep" />
        <span className="flex-1 bg-pine" />
      </div>

      <div className="mx-auto max-w-7xl px-6 py-14 md:px-10">
        <div className="grid grid-cols-2 gap-10 md:grid-cols-4">
          <div className="col-span-2 md:col-span-1">
            <Link
              href={ROUTES.HOME}
              className="inline-flex items-center gap-2.5 text-ink transition-opacity hover:opacity-75"
              aria-label="QuickPitik home"
            >
              <svg className="size-7" viewBox="0 0 28 28" fill="none" aria-hidden="true">
                <circle cx="14" cy="14" r="13" stroke="currentColor" strokeWidth="1.5" />
                <circle cx="14" cy="14" r="5" className="fill-fresh" />
              </svg>
              <span className="font-display text-lg font-extrabold tracking-tight">
                QuickPitik
              </span>
            </Link>
            <p className="mt-4 max-w-xs font-sans text-sm leading-relaxed text-slate">
              Race photos delivered minutes after the finish line. Find yours by
              face or bib in seconds — and photographers get found and paid.
            </p>
          </div>

          <FooterColumn title="Events">
            <FooterLink href={ROUTES.EVENTS}>Browse events</FooterLink>
            <FooterLink href="/#how-it-works">How it works</FooterLink>
          </FooterColumn>

          <FooterColumn title="Account">
            <FooterLink href={ROUTES.LOGIN}>Log in</FooterLink>
            <FooterLink href={ROUTES.REGISTER}>Sign up</FooterLink>
          </FooterColumn>

          <FooterColumn title="Support">
            <li>
              <a
                href="mailto:support@quickpitik.ph"
                className="nav-link font-sans text-sm text-ink-soft hover:text-fresh transition-colors"
              >
                support@quickpitik.ph
              </a>
            </li>
          </FooterColumn>
        </div>

        <div className="mt-12 border-t border-line pt-8 flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:justify-between">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            &copy; {new Date().getFullYear()} QuickPitik · All rights reserved
          </p>
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft">
            Cebu · Philippines
          </p>
        </div>
      </div>
    </footer>
  );
}

function FooterColumn({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="kicker">{title}</h3>
      <ul className="mt-4 space-y-3">{children}</ul>
    </div>
  );
}

function FooterLink({
  href,
  children,
}: {
  href: string;
  children: React.ReactNode;
}) {
  return (
    <li>
      <Link
        href={href}
        className="nav-link font-sans text-sm text-ink-soft hover:text-fresh transition-colors"
      >
        {children}
      </Link>
    </li>
  );
}
