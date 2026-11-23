import Link from "next/link";
import { Fragment } from "react";
import { getAdminSectionGuideByHref } from "@/lib/admin-section-guide";

// A page's own words from the guide: what is here, its look-alike pages, and
// the terms it uses. Never hidden at any width (a phone has no hover tips).
export function AdminSectionIntro({ href }: { href: string }) {
  const entry = getAdminSectionGuideByHref(href);
  if (!entry) return null;
  const instead = entry.instead ?? [];
  const terms = entry.terms ?? [];
  return (
    <>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">{entry.intro}</p>
      {instead.length > 0 && (
        <p className="font-sans text-sm text-ink-soft mt-2 max-w-xl">
          {instead.map((i, n) => (
            <Fragment key={i.href}>
              {n > 0 && " "}
              {i.when}, use{" "}
              <Link
                href={i.href}
                className="underline decoration-line-strong underline-offset-4 hover:decoration-fresh"
              >
                {i.label}
              </Link>
              .
            </Fragment>
          ))}
        </p>
      )}
      {terms.length > 0 && (
        <dl className="font-sans text-sm text-slate mt-2 max-w-xl space-y-1">
          {terms.map((t) => (
            <div key={t.term}>
              <dt className="inline">{t.term}:</dt> <dd className="inline">{t.meaning}</dd>
            </div>
          ))}
        </dl>
      )}
    </>
  );
}
