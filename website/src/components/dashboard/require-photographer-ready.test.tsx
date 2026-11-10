import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { RequirePhotographerReady } from "@/components/dashboard/require-photographer-ready";

vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => ({ kind: "identity", reviewState: "manual_review" }) }));
vi.mock("@/lib/photographer-verification-sync", () => ({ syncVerificationNow: vi.fn() }));

// Event creation stays on profile approval (AA-I15): an identity check that is
// still waiting blocks uploads only, never the create-event form.
it("identity kind still allows event create", () => {
  render(<RequirePhotographerReady><p>create event form</p></RequirePhotographerReady>);
  expect(screen.getByText("create event form")).toBeInTheDocument();
});
