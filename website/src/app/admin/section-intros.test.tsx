import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { ROUTES } from "@/lib/constants";
import { expectSectionIntro } from "@/test/admin-section-intro";
import AdminDisputesPage from "./disputes/page";
import AdminFlagsPage from "./flags/page";
import AdminRelevancePage from "./relevance/page";

// The three rail pages with no page test of their own: their header shows the guide intro.
vi.mock("next/navigation", () => ({ useRouter: () => ({ replace: vi.fn(), push: vi.fn() }) }));
const kpis = vi.hoisted(() => ({ value: null as Record<string, number> | null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));
const flagCounts = vi.hoisted(() => ({ open: null as number | null, hidden: null as number | null }));
vi.mock("@/hooks/use-admin-disputes", () => ({ useRefundedThisWeekTotal: () => undefined }));
vi.mock("@/components/admin/disputes-queue", () => ({ DisputesQueue: () => null }));
vi.mock("@/components/admin/flags-queue", () => ({
  FlagsQueue: () => null,
  useOpenFlagsCount: () => flagCounts.open,
  useHiddenFlagsCount: () => flagCounts.hidden,
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

afterEach(() => { kpis.value = null; flagCounts.open = null; flagCounts.hidden = null; });

const kicker = () => screen.getByRole("banner").querySelector("p");

it("the Disputes kicker counts what is waiting", () => {
  kpis.value = { disputesNeedingAdmin: 2 };
  render(<AdminDisputesPage />);
  expect(kicker()).toHaveTextContent(/^Disputes · 2 waiting · /);
});

it("the Flags kicker counts what is waiting and what is hidden", () => {
  flagCounts.open = 3; flagCounts.hidden = 1;
  render(<AdminFlagsPage />);
  expect(kicker()).toHaveTextContent(/^Flags · 3 waiting · 1 hidden$/);
});

it("the Photo Relevance kicker shows the waiting count once known, and the label alone before", () => {
  const view = render(<AdminRelevancePage />);
  expect(kicker()).toHaveTextContent(/^Photo Relevance$/);
  view.unmount();
  kpis.value = { relevanceNeedsPerson: 4 };
  render(<AdminRelevancePage />);
  expect(kicker()).toHaveTextContent(/^Photo Relevance · 4 waiting$/);
});
