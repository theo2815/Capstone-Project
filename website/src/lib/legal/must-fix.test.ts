import { describe, expect, it } from "vitest";
import { PRIVACY_POLICY } from "./privacy";
import { TERMS_OF_SERVICE } from "./terms";
import type { LegalDoc, LegalSection } from "./types";

// Legal review 2026-11-04 must-fix items M1, M3, M4 (task PT-MF).
const text = (sections: readonly LegalSection[]) =>
  sections
    .flatMap((s) => s.blocks)
    .flatMap((b) => (typeof b === "string" ? [b] : "list" in b ? b.list : b.table.flat()))
    .join("\n");
const section = (doc: LegalDoc, id: string) => text(doc.sections.filter((s) => s.id === id));

describe("legal review must-fix text", () => {
  it("does not promise a compulsory admin second sign-in step (M1)", () => {
    expect(text(PRIVACY_POLICY.sections)).not.toMatch(/Admins need|require a second sign-in step/);
  });

  it("declares the promotion use of event photos (M3)", () => {
    expect(section(PRIVACY_POLICY, "how-we-use-it")).toContain("**Promoting QuickPitik:**");
  });

  it("sets the shared-copyright rules instead of the Civil Code defaults (M4)", () => {
    const copyright = section(TERMS_OF_SERVICE, "copyright");
    expect(copyright).toContain("undivided half share");
    expect(copyright).toContain("10 years");
    expect(copyright).not.toContain("(G10)");
  });
});
