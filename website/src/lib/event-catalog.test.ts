import { describe, expect, it } from "vitest";
import {
  canOwnerUploadToEvent,
  canUploadToEvent,
  deriveEventState,
  ownerUploadCloseDate,
} from "./event-catalog";

// Race day is 2026-09-04 in Manila (UTC+8). The server may be UTC.
const RACE = "2026-09-04";

describe("deriveEventState is pinned to Asia/Manila", () => {
  it("is live from 00:00 PHT even while UTC is still the day before", () => {
    // 2026-09-03T16:30Z = 2026-09-04 00:30 PHT
    const now = new Date("2026-09-03T16:30:00Z");
    expect(deriveEventState(RACE, now)).toBe("live");
    expect(canUploadToEvent(RACE, now)).toBe(true);
  });

  it("is still upcoming one minute before Manila midnight", () => {
    // 2026-09-03T15:59Z = 2026-09-03 23:59 PHT
    const now = new Date("2026-09-03T15:59:00Z");
    expect(deriveEventState(RACE, now)).toBe("upcoming");
    expect(canUploadToEvent(RACE, now)).toBe(false);
  });

  it("closes uploads after the grace window", () => {
    // 2026-09-08 09:00 PHT — day 4
    const now = new Date("2026-09-08T01:00:00Z");
    expect(canUploadToEvent(RACE, now)).toBe(false);
  });
});

// Photographer-owned events (V46): the owner uploads from any time before the
// event through one calendar month after the date. The close date must mirror
// Java LocalDate.plusMonths(1) (end-of-month clamped) and the boundary is
// inclusive — both must agree with PhotoUploadService.gate on the backend, to
// the day, or the tile/gate copy promises a date the server won't honour.
describe("owner upload window mirrors the backend 1-month grace", () => {
  it("close date is one calendar month after, clamped to end-of-month like plusMonths", () => {
    expect(ownerUploadCloseDate("2026-06-15")).toBe("2026-07-15"); // plain
    expect(ownerUploadCloseDate("2026-01-31")).toBe("2026-02-28"); // Jan 31 -> Feb 28 (clamp)
    expect(ownerUploadCloseDate("2026-03-31")).toBe("2026-04-30"); // Mar 31 -> Apr 30 (clamp)
    expect(ownerUploadCloseDate("2026-12-15")).toBe("2027-01-15"); // year rollover
  });

  // Event on 2026-09-04 → uploads close at the end of 2026-10-04 (Manila).
  const OWN = "2026-09-04";

  it("has no lower bound — open well before the event date", () => {
    expect(canOwnerUploadToEvent(OWN, new Date("2026-08-01T00:00:00Z"))).toBe(true);
  });

  it("is still open on the close date, even while UTC has rolled to the next day", () => {
    // 2026-10-04T15:59Z = 2026-10-04 23:59 PHT — last minute of the window.
    expect(canOwnerUploadToEvent(OWN, new Date("2026-10-04T15:59:00Z"))).toBe(true);
    // 2026-10-03T16:30Z = 2026-10-04 00:30 PHT — first minute of the close date.
    expect(canOwnerUploadToEvent(OWN, new Date("2026-10-03T16:30:00Z"))).toBe(true);
  });

  it("closes the day after, the moment Manila ticks past the close date", () => {
    // 2026-10-04T16:30Z = 2026-10-05 00:30 PHT — one day past close.
    expect(canOwnerUploadToEvent(OWN, new Date("2026-10-04T16:30:00Z"))).toBe(false);
  });
});
