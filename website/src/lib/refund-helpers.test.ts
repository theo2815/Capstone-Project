import { describe, expect, it } from "vitest";
import type { RunnerDispute } from "@/lib/api-orders";
import type { OrderSummary } from "@/types/order";
import {
  getOrderRefundStatus,
  getRefundClaimLink,
  getRefundStatusLabel,
} from "./refund-helpers";

const dispute = (over: Partial<RunnerDispute> = {}): RunnerDispute => ({
  id: "d1",
  photoId: "p1",
  reason: "other",
  note: "",
  status: "open",
  resolution: null,
  refundAmount: null,
  resolutionNote: null,
  openedAt: "2026-10-20T00:00:00Z",
  resolvedAt: null,
  withdrawnAt: null,
  ...over,
});

const order = (disputes: RunnerDispute[]): OrderSummary =>
  ({ id: "o1", photoIds: ["p1"], disputes }) as unknown as OrderSummary;

describe("getRefundStatusLabel", () => {
  it("waits for the link when escalated + manual_review without a link", () => {
    expect(getRefundStatusLabel(dispute({ status: "escalated", refundStatus: "manual_review" }))).toBe(
      "Refund approved, waiting for PayMongo refund link",
    );
  });
  it("says the link was sent once a link exists", () => {
    expect(
      getRefundStatusLabel(
        dispute({ status: "escalated", refundStatus: "manual_review", refundLinkUrl: "https://pm.test/r" }),
      ),
    ).toBe("Refund link sent");
  });
  it("says the window closed when denied + cancelled", () => {
    expect(getRefundStatusLabel(dispute({ status: "denied", refundStatus: "cancelled" }))).toBe(
      "Refund cancelled: the refund window closed",
    );
  });
  it("returns null for plain open, escalated and resolved disputes", () => {
    expect(getRefundStatusLabel(dispute({ status: "open" }))).toBeNull();
    expect(getRefundStatusLabel(dispute({ status: "escalated" }))).toBeNull();
    expect(getRefundStatusLabel(dispute({ status: "resolved", refundStatus: "succeeded" }))).toBeNull();
  });
});

describe("getRefundClaimLink", () => {
  it("is null without a link", () => {
    expect(getRefundClaimLink(dispute({ status: "escalated", refundStatus: "manual_review" }))).toBeNull();
    expect(
      getRefundClaimLink(dispute({ status: "escalated", refundStatus: "manual_review", refundLinkUrl: "" })),
    ).toBeNull();
  });
  it("is null after the refund succeeded", () => {
    expect(
      getRefundClaimLink(dispute({ status: "resolved", refundStatus: "succeeded", refundLinkUrl: "https://pm.test/r" })),
    ).toBeNull();
  });
  it("is the url while manual_review", () => {
    expect(
      getRefundClaimLink(dispute({ status: "escalated", refundStatus: "manual_review", refundLinkUrl: "https://pm.test/r" })),
    ).toBe("https://pm.test/r");
  });
});

describe("getOrderRefundStatus label", () => {
  it("carries the label of the first non-withdrawn dispute that has one", () => {
    const status = getOrderRefundStatus(
      order([
        dispute({ id: "w", status: "withdrawn", refundStatus: "manual_review" }),
        dispute({ id: "d", status: "escalated", refundStatus: "manual_review" }),
      ]),
    );
    expect(status.label).toBe("Refund approved, waiting for PayMongo refund link");
    expect(status.kind).toBe("pending");
  });
  it("is null when no dispute has one", () => {
    expect(getOrderRefundStatus(order([dispute()])).label).toBeNull();
  });
});
