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

  it("FU-16: counts what the server counts: an uploaded cover and watermark, a payout account with its QR", () => {
    const payout = { id: "p-1", method: "gcash", accountNumber: "09175550101", accountName: "Ana", qr: null, isPrimary: true } as const;
    expect(view.snapshotFromSettings({
      brandName: "Studio", brandColor: "none", bio: "", handle: "studio", avatarUrl: null,
      cover: { kind: "gradient", from: "#000", to: "#111" }, watermark: { kind: "label", label: "Studio" },
      region: null, socials: [], payouts: [payout], source: "live",
    })).toMatchObject({ hasCover: false, hasWatermark: false, payoutCount: 0 });
    expect(view.snapshotFromSettings({
      brandName: "Studio", brandColor: "none", bio: "", handle: "studio", avatarUrl: null,
      cover: { kind: "image", url: "https://example.test/c.jpg" }, watermark: { kind: "image", dataUrl: "https://example.test/w.png" },
      region: null, socials: [], payouts: [{ ...payout, qr: { dataUrl: "https://example.test/q.png", uploadedAt: "2026-11-25T00:00:00Z" } }], source: "live",
    })).toMatchObject({ hasCover: true, hasWatermark: true, payoutCount: 1 });
  });
});
