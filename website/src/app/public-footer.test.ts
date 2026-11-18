import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, it } from "vitest";

// Every public page uses the one shared footer, not a private copy.
const PAGES = [
  "page.tsx",
  "runners/page.tsx",
  "photographers/page.tsx",
  "events/page.tsx",
  "events/[slug]/event-cockpit.tsx",
  "verify/page.tsx",
];

it.each(PAGES)("%s mounts the shared footer", (page) => {
  const source = readFileSync(path.join(__dirname, page), "utf8");
  expect(source).toContain('from "@/components/layout/footer"');
  expect(source).not.toMatch(/^function Footer\(/m);
  expect(source).not.toContain("<footer");
});

it.each(["runners/page.tsx", "photographers/page.tsx"])(
  "%s hides the footer CTA (it ends with its own)",
  (page) => {
    const source = readFileSync(path.join(__dirname, page), "utf8");
    expect(source).toContain("showCta={false}");
  },
);

it("event-cockpit keeps its footer clear of the sticky bars", () => {
  const source = readFileSync(path.join(__dirname, "events/[slug]/event-cockpit.tsx"), "utf8");
  expect(source.match(/<Footer className="pb-24 md:pb-20" \/>/g)).toHaveLength(3);
});

// A footer inside <main> is not exposed as a footer landmark, so it goes after it.
it.each(["page.tsx", "runners/page.tsx", "photographers/page.tsx", "events/page.tsx", "verify/page.tsx"])(
  "%s renders the footer after </main>",
  (page) => {
    const source = readFileSync(path.join(__dirname, page), "utf8");
    expect(source.indexOf("<Footer")).toBeGreaterThan(source.lastIndexOf("</main>"));
  },
);

// Sign-in, sign-up, password, email and legal pages all go through AuthShell.
it("AuthShell mounts the shared footer after </main>, not the old one-line footer", () => {
  const source = readFileSync(path.join(__dirname, "../components/auth/auth-shell.tsx"), "utf8");
  expect(source).toContain('from "@/components/layout/footer"');
  expect(source).not.toContain("PH &middot;");
  expect(source.indexOf("<Footer")).toBeGreaterThan(source.lastIndexOf("</main>"));
});

it("the photographer profile page ends with the same slim bar as its event pages", () => {
  const source = readFileSync(path.join(__dirname, "[handle]/page.tsx"), "utf8");
  expect(source).toContain("<ProfileShellFooter />");
});

it("events/page.tsx hides the footer CTA (the visitor is already on /events)", () => {
  const source = readFileSync(path.join(__dirname, "events/page.tsx"), "utf8");
  expect(source).toContain("showCta={false}");
});
