import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { PRIVACY_POLICY } from "@/lib/legal/privacy";
import { TERMS_OF_SERVICE } from "@/lib/legal/terms";
import { LegalDocument, LegalText } from "./legal-document";

describe("LegalText", () => {
  it("turns **bold** and [label](href) into markup", () => {
    const { container } = render(
      <p>
        <LegalText text="Read **this** and the [Privacy Policy](/privacy#deletion)." />
      </p>,
    );
    expect(container.querySelector("strong")?.textContent).toBe("this");
    expect(screen.getByRole("link", { name: "Privacy Policy" }).getAttribute("href")).toBe(
      "/privacy#deletion",
    );
    expect(container.textContent).toBe("Read this and the Privacy Policy.");
  });
});

describe("LegalDocument", () => {
  // Other pages and the mobile app link to these anchors.
  it.each([
    [PRIVACY_POLICY, ["deletion", "face-matching", "photo-metadata", "photos-of-you"]],
    [TERMS_OF_SERVICE, ["refunds", "prohibited-conduct", "buyer-licence", "copyright"]],
  ])("renders the section anchors other pages link to", (doc, ids) => {
    const { container } = render(<LegalDocument doc={doc} />);
    for (const id of ids) expect(container.querySelector(`section#${id}`)).not.toBeNull();
  });

  it("renders the refund policy inside the terms' refunds section", () => {
    const { container } = render(<LegalDocument doc={TERMS_OF_SERVICE} />);
    expect(container.querySelector("section#refunds")?.textContent).toContain("Accepted reasons");
  });

  it("words the refund rules for a standalone page, not a receipt", () => {
    const { container } = render(<LegalDocument doc={TERMS_OF_SERVICE} />);
    const text = container.querySelector("section#refunds")?.textContent ?? "";
    expect(text).toContain("your receipt");
    expect(text).not.toContain("this receipt");
  });
});
