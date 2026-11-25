import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { EffectivePhotographerSettings } from "@/lib/admin-photographer-view";
import {
  CompletenessSection,
  CoverSection,
  PayoutsSection,
  PublicUrlSection,
  SocialsSection,
  WatermarkSection,
} from "./photographer-review-sections";

const full: EffectivePhotographerSettings = {
  brandName: "Test Studio", brandColor: "none", bio: "", handle: "test-studio", avatarUrl: null,
  cover: { kind: "image", url: "https://cdn.example/cover.jpg" },
  watermark: { kind: "label", label: "TEST STUDIO" },
  region: { regionCode: "07", provinceCode: "0722" },
  socials: [{ id: "s1", platform: "instagram", url: "https://instagram.com/test" }],
  payouts: [{ id: "p1", method: "gcash", accountNumber: "09175550101", accountName: "Test Owner", qr: null, isPrimary: true }],
  source: "live",
};

afterEach(cleanup);

it("renders every section from one set of settings", () => {
  const { container } = render(
    <>
      <CoverSection effective={full} />
      <PublicUrlSection handle="test-studio" />
      <WatermarkSection effective={full} brand="Test Studio" />
      <SocialsSection effective={full} />
      <PayoutsSection effective={full} />
      <CompletenessSection snapshot={{ hasCover: true, hasBrandName: true, hasWatermark: true, hasHandle: true, hasRegion: false, socialCount: 1, payoutCount: 0 }} />
    </>,
  );
  expect(container.querySelector('img[src="https://cdn.example/cover.jpg"]')).not.toBeNull();
  expect(screen.getByText("quickpitik.com/test-studio")).toBeInTheDocument();
  for (const link of screen.getAllByRole("link", { name: "Open ↗" })) expect(link).toHaveClass("min-h-[44px]");
  expect(screen.getByLabelText("Watermark label: TEST STUDIO")).toBeInTheDocument();
  expect(screen.getByText("Instagram")).toBeInTheDocument();
  expect(screen.getByText("GCash")).toBeInTheDocument();
  expect(screen.getByText("Primary")).toBeInTheDocument();
  expect(screen.getAllByRole("listitem").slice(-7).map((li) => li.textContent)).toEqual([
    "Cover bannerset", "Brand nameset", "Watermarkset", "Public handleset", "Regionmissing", "Social profiles1 on file", "Payout accounts0 on file",
  ]);
});

it("links only safe web addresses, names the watermark image, and tags only the primary payout", () => {
  render(
    <>
      <SocialsSection effective={{ ...full, socials: [
        { id: "s1", platform: "instagram", url: "https://instagram.com/test" },
        { id: "s2", platform: "website", url: "javascript:alert(1)" },
      ] }} />
      <WatermarkSection effective={{ ...full, watermark: { kind: "image", dataUrl: "data:image/png;base64,AA" } }} brand="Test Studio" />
      <PayoutsSection effective={{ ...full, payouts: [{ ...full.payouts[0], isPrimary: false }] }} />
    </>,
  );
  const links = screen.getAllByRole("link", { name: "Open ↗" });
  expect(links).toHaveLength(1);
  expect(links[0]).toHaveAttribute("href", "https://instagram.com/test");
  expect(links[0]).toHaveAttribute("rel", "noopener noreferrer");
  // FU-18 (B-37): a 44 px tap target, like the public address's Open link.
  expect(links[0]).toHaveClass("min-h-[44px]");
  expect(screen.getByAltText("Test Studio watermark")).toBeInTheDocument();
  expect(screen.queryByText("Primary")).toBeNull();
});

it("says what is missing when nothing is on file", () => {
  render(
    <>
      <CoverSection effective={null} />
      <PublicUrlSection handle="" />
      <WatermarkSection effective={null} brand="Test Studio" />
      <SocialsSection effective={null} />
      <PayoutsSection effective={null} />
      <CompletenessSection snapshot={null} />
    </>,
  );
  for (const copy of ["No cover banner uploaded.", "No handle set yet.", "No watermark uploaded.", "No social profiles linked yet.", "No payout accounts on file.", "No settings on file."]) {
    expect(screen.getByText(copy)).toBeInTheDocument();
  }
});
