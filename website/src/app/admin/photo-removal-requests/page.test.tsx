import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import AdminPhotoRemovalRequestsPage from "./page";

vi.mock("@/components/admin/photo-removal/request-list", () => ({ RemovalRequestList: () => null }));

it("the heading is the rail label", () => {
  render(<AdminPhotoRemovalRequestsPage />);
  expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Photo removals.");
});
