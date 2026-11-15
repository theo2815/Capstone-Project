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
  it("passes the production guard with version 1.6", () => {
    expect(() =>
      assertPublishable(
        [JSON.stringify(PRIVACY_POLICY), JSON.stringify(TERMS_OF_SERVICE), JSON.stringify(LEGAL_COPY), LEGAL_LAST_UPDATED],
        LEGAL_VERSION,
        "production",
      ),
    ).not.toThrow();
    expect(LEGAL_VERSION).toBe("1.6");
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

  // The same notice sits on the profile library, which serves every event
  // (privacy browser pass finding 2, 2026-11-07); a clarification, so still v2.
  it("names no single event, so it reads right on the profile library too", () => {
    expect(LEGAL_COPY.faceNotice).toContain("with faces in the photos of the event you search.");
    expect(LEGAL_COPY.faceNotice).not.toContain("this event's photos");
  });

  it.each([
    ["privacy", "how-we-use-it", "Sections 12 and 13"],
    ["privacy", "how-we-use-it", "Automatic checks on uploads"],
    ["privacy", "how-we-use-it", "Safety, security, automatic checks on uploads and photo removal requests"],
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
    expect(privacy).toContain("an admin compares your photos by eye");
    expect(privacy).toContain("seller-check photos");
    expect(privacy).toContain("Photographer seller-check details and photos");
    expect(privacy).toContain("Seller-check details and photos you gave us are kept as legal records.");
  });

  it("no seller passage still says the checks are optional", () => {
    const all = [text(PRIVACY_POLICY.sections), text(TERMS_OF_SERVICE.sections)].join("\n");
    expect(all).not.toMatch(/Seller checks we may ask for|we may ask for your full name|if we asked for them/);
  });
});

// AA-I21: the adviser-signed 1.3 text (AA-0.7 draft, signed 2026-11-11). Its
// runner-deletion parts came in 1.4 with RD-11 (owner answer O1, RD-15 part B).
describe("legal 1.3 text (AA-I21)", () => {
  it("explains the seller checks: live selfie in Malaysia, ID number kept as a fingerprint", () => {
    const checks = section(PRIVACY_POLICY, "seller-checks");
    expect(checks).toContain("run by AWS in Malaysia");
    expect(checks).toContain("we keep only a fingerprint of it");
    expect(checks).toContain("Only an admin can reject them");
    expect(section(PRIVACY_POLICY, "what-we-collect")).not.toContain("we never run face recognition on them");
    expect(section(PRIVACY_POLICY, "who-receives-it")).toContain("the live checks in AWS's Malaysia region");
  });

  it("names the moderation checks and race events in the Terms", () => {
    expect(section(TERMS_OF_SERVICE, "photographers")).toContain("Details that pass every check are approved automatically");
    expect(section(TERMS_OF_SERVICE, "events-and-watermarks")).toContain("**Race events**");
    expect(section(TERMS_OF_SERVICE, "buying-photos")).toContain("up to twice the race price");
    expect(section(TERMS_OF_SERVICE, "what-quickpitik-may-do")).toContain("Buyers keep the photos they already bought.");
  });

  it("fills the identity copy with the signed text and links the seller checks", () => {
    for (const line of [LEGAL_COPY.identityNotice, LEGAL_COPY.identityTickLabel, LEGAL_COPY.identityAdminInstead]) {
      expect(line).not.toContain("[ASK OWNER");
    }
    expect(LEGAL_COPY.identityNotice).toContain("#seller-checks)");
  });

});

// RD-15 part B: the runner-deletion parts of the signed 1.3 draft, shipped as 1.4 (O1).
describe("legal 1.4 text (RD-15)", () => {
  it("explains the deletion live check, where it runs and how long the image is kept", () => {
    const face = section(PRIVACY_POLICY, "face-matching");
    expect(face).toContain("**Deletion requests with a live check.**");
    expect(face).toContain("When someone asks us by email to remove photos of them");
    expect(section(PRIVACY_POLICY, "who-receives-it")).toContain("the live checks (for seller checks and deletion requests) in Malaysia");
    expect(section(PRIVACY_POLICY, "how-long-we-keep-it")).toContain("Photo removal evidence, including the live-check image");
    expect(section(PRIVACY_POLICY, "photos-of-you")).toContain("**With a runner account:**");
    expect(section(PRIVACY_POLICY, "on-your-device")).toContain("(a seller check or a deletion request)");
    expect(section(TERMS_OF_SERVICE, "photo-removal")).toContain("runner accounts can also ask on the website with a live check");
  });

  it("fills the four runner deletion texts with the signed wording", () => {
    expect(JSON.stringify(LEGAL_COPY)).not.toContain("[ASK OWNER");
    expect(LEGAL_COPY.runnerDeletionMailSubject).toBe("Photo deletion request");
    expect(LEGAL_COPY.runnerDeletionTickLabel).toBe("I agree to the live check and to this face search. The photos I'm asking about are of me.");
    expect(LEGAL_COPY.runnerDeletionNotice).toMatch(/^To find photos of you, we run a short live check/);
    expect(LEGAL_COPY.runnerDeletionEmailInstead).toContain("subject=Photo%20deletion%20request)");
  });
});

// PE-2.5: the payout hold and the upload checks in the Terms and the Privacy Policy, version 1.5.
// Wording: PE-L4 texts T10 to T13, R-a and R-b (owner, 2026-11-12).
describe("legal 1.5 text (PE-2.5)", () => {
  it("lets the Terms put payouts on hold while uploads are reviewed, and says how the photographer is told", () => {
    const payouts = section(TERMS_OF_SERVICE, "photographers");
    expect(payouts).toContain("put your payouts on hold");
    expect(payouts).toContain("Your earnings stay in your balance.");
    expect(payouts).toContain("in your QuickPitik inbox when a hold starts and when it ends");
    expect(payouts).not.toContain("by email when it starts");
    expect(payouts).toContain("withhold 1% tax");
  });

  it("lists the hold among what QuickPitik may do, with the 'and' on the last item", () => {
    const mayDo = section(TERMS_OF_SERVICE, "what-quickpitik-may-do");
    expect(mayDo).toContain("hold a photographer's payouts");
    expect(mayDo).toContain("hide a race event from the Events page, or list it again;\n");
    expect(mayDo).not.toContain("list it again; and");
  });

  it("lists the hold e-mail and tells who sees the check results", () => {
    const how = section(PRIVACY_POLICY, "how-we-use-it");
    expect(how).toContain("an alert when a photographer's payouts are put on hold,");
    expect(how).toContain("we may send the photographer a message in their inbox");
    expect(how).toContain("an admin may see both photographers' names and sample photos side by side");
    expect(how).toContain("the photographer is told in their inbox when a hold starts and when it ends");
    expect(how).not.toContain("A person reviews them before any photo is hidden");
  });

  it("names upload checks among what admins see and keeps their results in the retention table", () => {
    expect(section(PRIVACY_POLICY, "who-receives-it")).toContain("reports, upload checks and removal requests");
    expect(section(PRIVACY_POLICY, "how-long-we-keep-it")).toContain(
      "Results of automatic checks on uploads, and admin reviews of them",
    );
  });
});

// RV-L3: what record-only adds, version 1.6. Wording: RV-L2 texts V2-T1 to V2-T4 (owner, 2026-11-15).
const rowOf = (doc: LegalDoc, id: string, first: string) =>
  doc.sections
    .filter((s) => s.id === id)
    .flatMap((s) => s.blocks)
    .flatMap((b) => (typeof b !== "string" && "table" in b ? b.table : []))
    .find((row) => row[0] === first);

describe("legal 1.6 text (RV-L3)", () => {
  it("says what the automatic checks add, and still that a person reviews before a hide (V2-T1)", () => {
    const how = section(PRIVACY_POLICY, "how-we-use-it");
    expect(how).toContain(
      "compare the date the camera recorded with the event's date. For photos in Public events, we also check how sharp and how bright the photos are and what they show (for example a person or a race), on a sample of 300 of each photographer's photos in an event, or up to 600 when the first result is close to a limit, using Amazon Rekognition. We count the words printed in the photos, such as on banners, shirts and signs, as the number of photos each word appears on, not which photo it is in. We then send a short text summary of these results to Anthropic, whose AI model Claude suggests whether the photos fit the event. Anthropic never receives the photos. We do not send the photographer's name or account details; the event's name and place are sent as the event shows them. We store the results",
    );
    expect(how).toContain("A person reviews the results before any photo is hidden");
  });

  it("names Anthropic as a receiver, where it processes the summaries and how long it keeps them (V2-T2)", () => {
    expect(rowOf(PRIVACY_POLICY, "who-receives-it", "Anthropic")).toEqual([
      "Anthropic",
      "a short text summary of one photographer's photos in one event: the event's name, date and place, counts from our photo checks, the results of our copy and date checks, the most common words printed in the photos with how many photos show each, and how many other photographers in the event have the same words. No photos. No photographer's name or account details; the event's name and place are sent as the event shows them",
      "checking whether a photographer's photos fit the event, with its AI model Claude",
    ]);
    expect(section(PRIVACY_POLICY, "who-receives-it")).toContain(
      "comparable with Philippine law. Anthropic processes the summaries in the United States, Europe, Asia or Australia, stores them in the United States, and deletes them within 30 days, or keeps them for up to 2 years if its safety systems flag them, under its standard commercial terms.",
    );
  });

  it("says Rekognition also reads other words and checks sharpness, brightness and content (V2-T3)", () => {
    expect(rowOf(PRIVACY_POLICY, "who-receives-it", "Amazon Web Services (Amazon Rekognition)")?.[2]).toBe(
      "face matching, reading bib numbers and other words printed in event photos, checking event photos for sharpness, brightness and what they show, ID text reading, seller face comparison and image checks in AWS's Singapore region; the live checks in AWS's Malaysia region",
    );
  });

  it("keeps rare words for 7 days after the checks finish, and says so above the table (V2-T4)", () => {
    expect(
      rowOf(
        PRIVACY_POLICY,
        "how-long-we-keep-it",
        "Words read from event photos (counted for each photographer and event, not for each photo), photo measurements (sharpness, brightness and what a photo shows), and the summary sent to Anthropic with its answer",
      )?.[1],
    ).toBe(
      "measurements: until the photo, the event or the photographer's account is deleted. Words found on fewer than 3% of a photographer's photos in an event: deleted 7 days after our checks of those photos finish. Other words, the summary and the answer: until the event or the photographer's account is deleted. Deleting one photo does not change the word counts, because we do not record which photo a word came from.",
    );
    const keep = section(PRIVACY_POLICY, "how-long-we-keep-it");
    expect(keep).toContain(
      "Apart from face templates, which are deleted 6 months after the event, and rare words read from event photos (see the table below), we do not yet delete data automatically after a set time. Data is kept until you or QuickPitik delete it, as below.",
    );
    expect(text(PRIVACY_POLICY.sections)).not.toContain("[ASK OWNER");
  });
});
