import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunnerDispute } from "@/lib/api-orders";
import type { OrderSummary } from "@/types/order";
import {
  getOrderRefundStatus,
  getRefundClaimLink,
  getRefundStatusLabel,
  isRefundLinkExpired,
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

describe("expired refund link", () => {
  const expiry = Date.parse("2026-10-30T12:00:00Z");
  const sent = (expiresAt: string | null) =>
    dispute({
      status: "escalated",
      refundStatus: "manual_review",
      refundLinkUrl: "https://pm.test/r",
      refundLinkExpiresAt: expiresAt,
    });

  afterEach(() => vi.useRealTimers());

  it("isRefundLinkExpired is true only at or after the expiry instant", () => {
    const at = "2026-10-30T12:00:00Z";
    expect(isRefundLinkExpired(at, expiry - 1)).toBe(false);
    expect(isRefundLinkExpired(at, expiry)).toBe(true);
    expect(isRefundLinkExpired(at, expiry + 1)).toBe(true);
    expect(isRefundLinkExpired(null, expiry)).toBe(false);
    expect(isRefundLinkExpired(undefined, expiry)).toBe(false);
    expect(isRefundLinkExpired("", expiry)).toBe(false);
    expect(isRefundLinkExpired("garbage", expiry)).toBe(false);
  });
  it("keeps the claim link 1 ms before expiry, drops it at and after expiry", () => {
    const d = sent("2026-10-30T12:00:00Z");
    expect(getRefundClaimLink(d, expiry - 1)).toBe("https://pm.test/r");
    expect(getRefundClaimLink(d, expiry)).toBeNull();
    expect(getRefundClaimLink(d, expiry + 1)).toBeNull();
  });
  it("keeps the claim link when the expiry is missing or garbage", () => {
    expect(getRefundClaimLink(sent(null), expiry)).toBe("https://pm.test/r");
    expect(getRefundClaimLink(sent("garbage"), expiry)).toBe("https://pm.test/r");
  });
  it("labels the dispute expired or sent", () => {
    const d = sent("2026-10-30T12:00:00Z");
    expect(getRefundStatusLabel(d, expiry)).toBe("Refund link expired");
    expect(getRefundStatusLabel(d, expiry - 1)).toBe("Refund link sent");
  });
  it("rolls up the expired label from a later dispute", () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(expiry + 1000);
    const o = {
      id: "o1",
      photoIds: ["p1", "p2"],
      disputes: [dispute({ id: "d1", photoId: "p1" }), { ...sent("2026-10-30T12:00:00Z"), id: "d2", photoId: "p2" }],
    } as unknown as OrderSummary;
    expect(getOrderRefundStatus(o).label).toBe("Refund link expired");
  });
});
