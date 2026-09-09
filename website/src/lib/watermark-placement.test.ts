import { describe, it, expect } from "vitest";
import { watermarkBox, positionFromBox } from "./watermark-placement";

describe("watermarkBox", () => {
  // Same numeric cases in backend WatermarkServiceTest and mobile
  // WatermarkPlacementMathTest. Dimensions are rendered-photo pixels.
  it.each([
    [400, 300, 3, 0.15, 1, 1, { left: 328, top: 268, width: 60, height: 20 }],
    [800, 1200, 3, 0.05, 0, 0, { left: 20, top: 20, width: 48, height: 16 }],
    [333, 222, 3, 0.157, 0.33, 0.73, { left: 97, top: 144, width: 52, height: 17 }],
    [1280, 853, 0.5, 0.4, 1, 1, { left: 854, top: 32, width: 394, height: 789 }],
    [40, 20, 0.5, 0.4, 1, 1, { left: 30, top: 9, width: 1, height: 2 }],
  ] as const)("matches backend geometry on a %i x %i photo", (w, h, aspect, scale, x, y, expected) => {
    expect(watermarkBox(w, h, aspect, scale, x, y)).toEqual(expected);
  });

  it("round-trips the rendered position through the drag inverse, including fitted logos", () => {
    for (const aspect of [0.5, 3, 100]) {
      const box = watermarkBox(1280, 853, aspect, 0.4, 0.3, 0.7);
      const pos = positionFromBox(1280, 853, aspect, 0.4, box.left, box.top, 0.3, 0.7);
      expect(watermarkBox(1280, 853, aspect, 0.4, pos.x, pos.y)).toEqual(box);
    }
  });

  it("clamps a drag at both edges and handles a logo that fills the available height", () => {
    expect(positionFromBox(1280, 853, 0.5, 0.4, -100, -100, 1, 0.7)).toEqual({ x: 0, y: 0.7 });
    expect(positionFromBox(1280, 853, 0.5, 0.4, 2000, 2000, 1, 0.7)).toEqual({ x: 1, y: 0.7 });
    expect(watermarkBox(400, 300, 3, 0.15, 5, -2)).toEqual({ left: 328, top: 12, width: 60, height: 20 });
  });
});
