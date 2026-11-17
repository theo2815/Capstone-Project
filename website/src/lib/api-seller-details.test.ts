import { expect, it } from "vitest";
import { formatGateDate, identityNoticeDate, type SellerDetails } from "./api-seller-details";

const details = (reviewState: SellerDetails["reviewState"], identityGateStartsOn: string | null): SellerDetails => ({
  status: "submitted", reviewState, identityGateStartsOn,
});

it("the notice shows through the gate date in Manila, then stops", () => {
  const d = details("manual_review", "2026-12-07");
  expect(identityNoticeDate(d, new Date("2026-12-07T15:59:00Z"))).toBe("2026-12-07"); // 23:59 on the 7th in Manila
  expect(identityNoticeDate(d, new Date("2026-12-07T16:00:00Z"))).toBeNull(); // 00:00 on the 8th in Manila
});

it("missing and rejected records get the notice; passed ones do not", () => {
  const now = new Date("2026-12-01T00:00:00Z");
  expect(identityNoticeDate(details("missing", "2026-12-07"), now)).toBe("2026-12-07");
  expect(identityNoticeDate(details("rejected", "2026-12-07"), now)).toBe("2026-12-07");
  expect(identityNoticeDate(details("approved", "2026-12-07"), now)).toBeNull();
  expect(identityNoticeDate(undefined, now)).toBeNull();
});

it("writes the gate date the way the notice does", () => {
  expect(formatGateDate("2026-12-07")).toBe("Monday 7 December 2026");
});
