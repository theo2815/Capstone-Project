import { render, screen } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import AdminPhotoRemovalRequestsPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("@/components/admin/photo-removal/request-list", () => ({ RemovalRequestList: () => null }));
const kpis = vi.hoisted(() => ({ value: null as Record<string, number> | null }));
vi.mock("@/hooks/use-admin-data", () => ({ useAdminKpis: () => kpis.value }));
afterEach(() => { kpis.value = null; });

it("the heading is the rail label", () => {
  render(<AdminPhotoRemovalRequestsPage />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Photo removals.");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  render(<AdminPhotoRemovalRequestsPage />);
  expectSectionIntro(ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS);
});

it("the kicker shows the waiting count once known, and the label alone before", () => {
  const view = render(<AdminPhotoRemovalRequestsPage />);
  expect(screen.getByRole("banner").querySelector("p")).toHaveTextContent(/^Photo removals$/);
  view.unmount();
  kpis.value = { removalsNeedingAction: 3 };
  render(<AdminPhotoRemovalRequestsPage />);
  expect(screen.getByRole("banner").querySelector("p")).toHaveTextContent(/^Photo removals · 3 waiting$/);
});
