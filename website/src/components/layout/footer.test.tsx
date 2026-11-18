import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import { Footer } from "@/components/layout/footer";
import { SUPPORT_EMAIL } from "@/lib/constants";

it("shows the CTA band and the footer links", () => {
  const { container } = render(<Footer />);
  expect(screen.getByRole("contentinfo")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Find my photos" })).toHaveAttribute("href", "/events");
  expect(screen.getByRole("heading", { name: "Find your finish." })).toBeInTheDocument();

  const links: [string, string][] = [
    ["For runners", "/runners"],
    ["Browse events", "/events"],
    ["Verify a photo", "/verify"],
    ["For photographers", "/photographers"],
    ["Join as a photographer", "/register?role=PHOTOGRAPHER"],
    [SUPPORT_EMAIL, `mailto:${SUPPORT_EMAIL}`],
    ["QuickPitik home", "/runners"],
  ];
  for (const [name, href] of links) {
    expect(screen.getByRole("link", { name })).toHaveAttribute("href", href);
  }
  expect(screen.getAllByRole("link", { name: /privacy/i })).toHaveLength(1);
  expect(screen.getAllByRole("link", { name: /terms/i })).toHaveLength(1);
  expect(screen.queryByRole("link", { name: /log in|sign up/i })).toBeNull();

  const hidden = [...container.querySelectorAll('[aria-hidden="true"]')];
  expect(hidden.some((el) => el.textContent === "QuickPitik")).toBe(true);
  expect(screen.getByText(`© ${new Date().getFullYear()} QuickPitik`)).toBeInTheDocument();
});

it("drops the CTA band with showCta={false}", () => {
  render(<Footer showCta={false} />);
  expect(screen.queryByRole("link", { name: "Find my photos" })).toBeNull();
  expect(screen.queryByRole("heading", { name: "Find your finish." })).toBeNull();
});
