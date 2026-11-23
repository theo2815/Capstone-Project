import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import AdminPhotoRemovalRequestsPage from "./page";
import { expectSectionIntro } from "@/test/admin-section-intro";
import { ROUTES } from "@/lib/constants";

vi.mock("@/components/admin/photo-removal/request-list", () => ({ RemovalRequestList: () => null }));

it("the heading is the rail label", () => {
  render(<AdminPhotoRemovalRequestsPage />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Photo removals.");
});

it("shows its guide intro, look-alike links and definitions without hover", () => {
  render(<AdminPhotoRemovalRequestsPage />);
  expectSectionIntro(ROUTES.ADMIN_PHOTO_REMOVAL_REQUESTS);
});
