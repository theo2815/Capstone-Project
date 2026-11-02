import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LegalLinks } from "./legal-links";

describe("LegalLinks", () => {
  it("links to the privacy policy and the terms", () => {
    render(<LegalLinks />);
    expect(screen.getByRole("link", { name: "Privacy" }).getAttribute("href")).toBe("/privacy");
    expect(screen.getByRole("link", { name: "Terms" }).getAttribute("href")).toBe("/terms");
  });
});
