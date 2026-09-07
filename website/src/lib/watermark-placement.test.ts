import { describe, it, expect } from "vitest";
import { watermarkBox, positionFromBox } from "./watermark-placement";

// The preview must land the logo where the backend bakes it. These lock the
// free-space formula that both sides share.
describe("watermarkBox", () => {
  it("default (1,1) lands bottom-right with the edge pad", () => {
    // 3:1 logo, 15% scale, on a 400×300 box.
    const b = watermarkBox(400, 300, 3, 0.15, 1, 1);
    const pad = 400 * 0.025; // 10
    const width = 60; // 400 * 0.15
    const height = 20; // 60 / 3
    expect(b.width).toBeCloseTo(width);
    expect(b.height).toBeCloseTo(height);
    expect(b.left).toBeCloseTo(400 - width - pad); // 330
    expect(b.top).toBeCloseTo(300 - height - pad); // 270
  });

  it("(0,0) lands top-left with the edge pad", () => {
    const b = watermarkBox(400, 300, 3, 0.15, 0, 0);
    expect(b.left).toBeCloseTo(10);
    expect(b.top).toBeCloseTo(10);
  });

  it("preserves the logo aspect ratio at any scale", () => {
    const b = watermarkBox(400, 300, 2, 0.25, 0.5, 0.5);
    expect(b.width / b.height).toBeCloseTo(2);
  });

  it("round-trips through positionFromBox (the drag inverse)", () => {
    const b = watermarkBox(400, 300, 3, 0.2, 0.3, 0.7);
    const p = positionFromBox(400, 300, 3, 0.2, b.left, b.top);
    expect(p.x).toBeCloseTo(0.3);
    expect(p.y).toBeCloseTo(0.7);
  });

  it("clamps positions into the frame", () => {
    const b = watermarkBox(400, 300, 3, 0.15, 5, -2);
    expect(b.left).toBeCloseTo(400 - 60 - 10); // clamped to bottom-right
    expect(b.top).toBeCloseTo(10); // clamped to top
  });
});
