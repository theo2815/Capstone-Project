import { describe, expect, it } from "vitest";
import { FACE_NOTICE_VERSION, LEGAL_COPY } from "./copy";
import { PRIVACY_POLICY } from "./privacy";
import { assertPublishable } from "./publishable";
import { TERMS_OF_SERVICE } from "./terms";
import type { LegalDoc, LegalSection } from "./types";
import { LEGAL_LAST_UPDATED, LEGAL_VERSION } from "./version";

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

// PT-0b (owner's answers 2026-11-04): the review's should-fix wording applied,
// so a Vercel production build passes the guard. Version 1.1 since BT-1.
describe("approved text (PT-0b)", () => {
  it("passes the production guard with version 1.2", () => {
    expect(() =>
      assertPublishable(
        [JSON.stringify(PRIVACY_POLICY), JSON.stringify(TERMS_OF_SERVICE), JSON.stringify(LEGAL_COPY), LEGAL_LAST_UPDATED],
        LEGAL_VERSION,
        "production",
      ),
    ).not.toThrow();
    expect(LEGAL_VERSION).toBe("1.2");
  });

  it("says both selfie and bib searches ask for a tick first (BT-1, S10)", () => {
    expect(section(PRIVACY_POLICY, "face-matching")).toContain("**Before you search** by selfie or by bib number");
    expect(section(PRIVACY_POLICY, "face-matching")).not.toContain("Before your first search");
  });

  it("says face templates are deleted 6 months after the event (M2)", () => {
    expect(section(PRIVACY_POLICY, "face-matching")).toContain("deleted 6 months after the event date");
    expect(section(PRIVACY_POLICY, "how-long-we-keep-it")).toContain("6 months after the event date");
  });

  it("asks every device again after the face notice changed (S10)", () => {
    expect(LEGAL_COPY.faceNotice).toContain("Singapore");
    expect(LEGAL_COPY.faceTickLabel).toContain("parent or guardian");
    expect(FACE_NOTICE_VERSION).toBe(2);
  });

  it.each([
    ["privacy", "how-we-use-it", "Sections 12 and 13"],
    ["privacy", "face-matching", "You can object"],
    ["privacy", "who-receives-it", "QuickPitik stays responsible"],
    ["privacy", "what-we-collect", "technical logs may record it"],
    ["privacy", "what-we-collect", "BIR Certificate of Registration"],
    ["privacy", "your-rights", "We reply within 15 days"],
    ["privacy", "how-long-we-keep-it", "last year recorded in our books"],
    ["privacy", "photos-of-you", "72 hours"],
    ["terms", "photographers", "Internet Transactions Act (RA 11967)"],
    ["terms", "photographers", "withhold 1% tax"],
    ["terms", "refunds", "Philippine consumer law"],
    ["terms", "appeals", "National Privacy Commission"],
    ["terms", "photo-removal", "72 hours"],
    ["terms", "liability", "Consumer Act (RA 7394)"],
    ["terms", "governing-law", "any other court"],
  ])("%s %s carries the review wording: %s", (doc, id, phrase) => {
    expect(section(doc === "privacy" ? PRIVACY_POLICY : TERMS_OF_SERVICE, id)).toContain(phrase);
  });
});

// SC-2a (owner, 2026-11-05): seller checks are required before uploading, version 1.2.
describe("seller checks text (SC-2a)", () => {
  it("says the seller checks are required before uploading", () => {
    expect(section(TERMS_OF_SERVICE, "photographers")).toContain("Before you can upload photos, you must give us your full legal name");
    expect(section(PRIVACY_POLICY, "what-we-collect")).toContain("Seller checks, required before you can upload");
  });

  it("says admins compare by eye and names the seller-check photos in storage", () => {
    const privacy = text(PRIVACY_POLICY.sections);
    expect(privacy).toContain("we never run face recognition on them");
    expect(privacy).toContain("seller-check photos");
    expect(privacy).toContain("Photographer seller-check details and photos");
    expect(privacy).toContain("Seller-check details and photos you gave us are kept as legal records.");
  });

  it("no seller passage still says the checks are optional", () => {
    const all = [text(PRIVACY_POLICY.sections), text(TERMS_OF_SERVICE.sections)].join("\n");
    expect(all).not.toMatch(/Seller checks we may ask for|we may ask for your full name|if we asked for them/);
  });
});
