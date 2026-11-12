import { render, screen } from "@testing-library/react";
import { expect, it } from "vitest";
import type { Dispute } from "@/lib/admin-disputes";
import { dispute } from "@/test/fixtures/admin";
import { AdminDisputeCard } from "./admin-dispute-card";

it("an open dispute shows the photo's own charge, not the order total", () => {
  const bundled: Dispute = { ...dispute, itemChargeAmount: 127.5, orderSnapshot: { ...dispute.orderSnapshot, total: 300 } };
  render(<AdminDisputeCard dispute={bundled} onOpen={() => {}} />);
  const card = screen.getByRole("button", { name: `Open dispute ${dispute.id}` });
  expect(card).toHaveTextContent("₱127.50");
  expect(card).not.toHaveTextContent("₱300.00");
});

it("a QR Ph payment is labelled QR Ph", () => {
  const qrph: Dispute = { ...dispute, orderSnapshot: { ...dispute.orderSnapshot, paymentMethod: "qrph" } };
  render(<AdminDisputeCard dispute={qrph} onOpen={() => {}} />);
  expect(screen.getByRole("button", { name: `Open dispute ${dispute.id}` })).toHaveTextContent("QR Ph");
});
