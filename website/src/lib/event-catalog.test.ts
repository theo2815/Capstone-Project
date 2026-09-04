import { describe, expect, it } from "vitest";
import { canUploadToEvent, deriveEventState } from "./event-catalog";

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
