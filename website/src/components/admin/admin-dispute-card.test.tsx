import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import type { Dispute } from "@/lib/admin-disputes";
import { dispute } from "@/test/fixtures/admin";
import { AdminDisputeCard } from "./admin-dispute-card";

afterEach(cleanup);
const title = `@${dispute.runnerHandle} vs @${dispute.photographerHandle}`;

it("an open dispute shows the photo's own charge, not the order total", () => {
  const bundled: Dispute = { ...dispute, itemChargeAmount: 127.5, orderSnapshot: { ...dispute.orderSnapshot, total: 300 } };
  const { container } = render(<AdminDisputeCard dispute={bundled} onOpen={() => {}} />);
  expect(container).toHaveTextContent("₱127.50");
  expect(container).not.toHaveTextContent("₱300.00");
});

it("a QR Ph payment is labelled QR Ph", () => {
  const qrph: Dispute = { ...dispute, orderSnapshot: { ...dispute.orderSnapshot, paymentMethod: "qrph" } };
  const { container } = render(<AdminDisputeCard dispute={qrph} onOpen={() => {}} />);
  expect(container).toHaveTextContent("QR Ph");
});

it("the card opens through a button named by its title, and the heading is not inside a button", async () => {
  const onOpen = vi.fn();
  render(<AdminDisputeCard dispute={dispute} onOpen={onOpen} />);
  await userEvent.click(screen.getByRole("button", { name: title }));
  expect(onOpen).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("heading", { name: title }).closest("button")).toBeNull();
});
