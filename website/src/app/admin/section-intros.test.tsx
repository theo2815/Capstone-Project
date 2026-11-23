import { render } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ROUTES } from "@/lib/constants";
import { expectSectionIntro } from "@/test/admin-section-intro";
import AdminDisputesPage from "./disputes/page";
import AdminFlagsPage from "./flags/page";
import AdminRelevancePage from "./relevance/page";

// The three rail pages with no page test of their own: their header shows the guide intro.
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: vi.fn(), push: vi.fn() }) }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => null }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useRefundedThisWeekTotal: () => undefined }));
vi.mock("@/components/admin/disputes-queue", () => ({ DisputesQueue: () => null }));
vi.mock("@/components/admin/flags-queue", () => ({
  FlagsQueue: () => null,
  useOpenFlagsCount: () => null,
  useHiddenFlagsCount: () => null,
}));
vi.mock("@/components/admin/relevance/relevance-cases", () => ({ RelevanceCases: () => null }));

it("Disputes shows its guide intro", () => {
  render(<AdminDisputesPage />);
  expectSectionIntro(ROUTES.ADMIN_DISPUTES);
});

it("Flags shows its guide intro and links to Photo Relevance and Photo removals", () => {
  render(<AdminFlagsPage />);
  expectSectionIntro(ROUTES.ADMIN_FLAGS);
});

it("Photo Relevance shows its guide intro and links to Flags and Photo removals", () => {
  render(<AdminRelevancePage />);
  expectSectionIntro(ROUTES.ADMIN_RELEVANCE);
  expect(document.body).not.toHaveTextContent("Uploaded batches checked");
});
