import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import { AdminPayoutReasonModal } from "./admin-payout-reason-modal";

afterEach(cleanup);

// FU-9 (U9.1, G6): Mark failed says the photographer gets the reason.
it("Mark failed says the money goes back and the photographer gets the reason", () => {
  render(<AdminPayoutReasonModal open kind="failed" payoutId="cycle-1" amount={1200} onClose={() => {}} onSubmit={() => {}} />);
  expect(screen.getByRole("dialog")).toHaveTextContent(
    "Use this when the money never left. The money goes back to the photographer's balance, and they get your reason in their QuickPitik inbox.",
  );
});
