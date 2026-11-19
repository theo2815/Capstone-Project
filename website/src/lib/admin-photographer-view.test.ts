import { describe, expect, it } from "vitest";
import * as view from "./admin-photographer-view";
import type { PhotographerSettingsSnapshot } from "./admin-user-registry";

const noLinks: PhotographerSettingsSnapshot = {
  hasCover: true, hasBrandName: true, hasWatermark: true, hasHandle: true, hasRegion: true, socialCount: 0, payoutCount: 0,
};

describe("computeCompleteness", () => {
  it("counts the five settings plus socials and payouts, out of seven", () => {
    expect(view.computeCompleteness(noLinks)).toEqual({ filled: 5, total: 7 });
    expect(view.computeCompleteness({ ...noLinks, socialCount: 2, payoutCount: 1 })).toEqual({ filled: 7, total: 7 });
    expect(view.computeCompleteness(null)).toEqual({ filled: 0, total: 7 });
  });

  it("lists the same seven items it counts", () => {
    const items = view.completenessItems(noLinks);
    expect(items.map((i) => i.label)).toEqual([
      "Cover banner", "Brand name", "Watermark", "Public handle", "Region", "Social profiles", "Payout accounts",
    ]);
    expect(items.filter((i) => i.filled)).toHaveLength(view.computeCompleteness(noLinks).filled);
  });

  it("derives the snapshot from live settings (a handle needs three letters)", () => {
    expect(view.snapshotFromSettings({
      brandName: " ", brandColor: "none", bio: "", handle: "ab", avatarUrl: null, cover: null, watermark: null,
      region: null, socials: [], payouts: [], source: "live",
    })).toEqual({ hasCover: false, hasBrandName: false, hasWatermark: false, hasHandle: false, hasRegion: false, socialCount: 0, payoutCount: 0 });
  });
});
