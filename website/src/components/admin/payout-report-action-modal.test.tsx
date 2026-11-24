import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { PayoutReport } from "@/lib/admin-payout-reports";
import { PayoutReportActionModal } from "./payout-report-action-modal";

afterEach(cleanup);

const report = { id: "report-1", handle: "studio" } as PayoutReport;
const LINE = "The photographer gets your reply in their QuickPitik inbox. A reply cannot be taken back; you can still resolve the report later.";

// FU-9 (U9.3): Acknowledge says who reads the reply and what can still change.
it("Acknowledge says the reply cannot be taken back; Resolve does not", () => {
  render(<PayoutReportActionModal open mode="acknowledge" report={report} onClose={() => {}} onSubmit={() => {}} />);
  expect(screen.getByRole("dialog")).toHaveTextContent(LINE);
  cleanup();
  render(<PayoutReportActionModal open mode="resolve" report={report} onClose={() => {}} onSubmit={() => {}} />);
  expect(screen.getByRole("dialog")).not.toHaveTextContent(LINE);
});
