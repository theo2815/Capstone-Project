import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { VerificationBanner } from "./verification-banner";

const { seller } = vi.hoisted(() => ({ seller: { data: undefined as unknown } }));
vi.mock("next/navigation", () => ({ usePathname: () => "/dashboard" }));
vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "ok" }) }));
vi.mock("@/hooks/use-seller-details", () => ({ useSellerDetails: () => ({ query: seller }) }));

const details = (reviewState: string, identityGateStartsOn: string | null) => ({
  status: reviewState === "manual_review" ? "submitted" : reviewState, reviewState, uploadsAllowed: true, identityGateStartsOn,
});
const notice = () => screen.queryByText("A quick live selfie before you upload");

afterEach(() => { seller.data = undefined; });

it("shows the dated notice when set and not passed", () => {
  seller.data = details("manual_review", "2099-01-05");
  render(<VerificationBanner />);
  expect(notice()).toBeInTheDocument();
  expect(screen.getByText(/From Monday 5 January 2099, you need a passed identity check/)).toBeInTheDocument();
  expect(screen.getByRole("link", { name: /Do it now/ })).toHaveAttribute("href", "/dashboard/settings#seller-details");
});

it("hidden when passed or no date", () => {
  seller.data = details("auto_approved", "2099-01-05");
  const { rerender } = render(<VerificationBanner />);
  expect(notice()).not.toBeInTheDocument();
  seller.data = details("approved", "2099-01-05");
  rerender(<VerificationBanner />);
  expect(notice()).not.toBeInTheDocument();
  seller.data = details("manual_review", null);
  rerender(<VerificationBanner />);
  expect(notice()).not.toBeInTheDocument();
});

it("hidden once the date has passed", () => {
  seller.data = details("manual_review", "2020-01-06");
  render(<VerificationBanner />);
  expect(notice()).not.toBeInTheDocument();
});
