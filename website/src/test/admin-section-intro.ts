import { screen } from "@testing-library/react";
import { expect } from "vitest";
import { getAdminSectionGuideByHref } from "@/lib/admin-section-guide";

// A rail page shows its guide intro, a link to each look-alike page and each
// definition, with no hover (FT-9).
export function expectSectionIntro(href: string) {
  const entry = getAdminSectionGuideByHref(href)!;
  expect(screen.getByText(entry.intro)).toBeInTheDocument();
  for (const i of entry.instead ?? []) {
    const hrefs = screen.getAllByRole("link", { name: i.label }).map((a) => a.getAttribute("href"));
    expect(hrefs).toContain(i.href);
  }
  for (const t of entry.terms ?? []) expect(screen.getByText(t.meaning)).toBeInTheDocument();
}
