import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { scaleToFit, shrinkSelfie } from "./image-utils";

it.each([
  [6000, 4000, 1280, 853],
  [3000, 4000, 960, 1280],
  [800, 600, 800, 600],
  [1280, 1280, 1280, 1280],
])("scales %ix%i to fit 1280 as %ix%i without upscaling", (w, h, width, height) => {
  expect(scaleToFit(w, h, 1280)).toEqual({ width, height });
});

// jsdom has no image decoder or canvas: fake both so the branches run.
let decoded: { width: number; height: number } | null;
const drawImage = vi.fn();
beforeEach(() => {
  decoded = null;
  drawImage.mockClear();
  vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:selfie");
  vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
  vi.stubGlobal("Image", class {
    width = 0;
    height = 0;
    onload: (() => void) | null = null;
    onerror: (() => void) | null = null;
    set src(_: string) {
      setTimeout(() => {
        if (!decoded) return this.onerror?.();
        Object.assign(this, decoded);
        this.onload?.();
      });
    }
  });
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({ drawImage } as unknown as CanvasRenderingContext2D);
  vi.spyOn(HTMLCanvasElement.prototype, "toBlob").mockImplementation((cb, type) => cb(new Blob(["jpeg"], { type })));
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("re-encodes a large photo as a 1280 px JPEG", async () => {
  decoded = { width: 4000, height: 3000 };
  const out = await shrinkSelfie(new File(["big"], "IMG_1.png", { type: "image/png" }));
  expect(out).toMatchObject({ name: "selfie.jpg", type: "image/jpeg" });
  expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 1280, 960);
  expect(HTMLCanvasElement.prototype.toBlob).toHaveBeenCalledWith(expect.any(Function), "image/jpeg", 0.85);
});

it("re-encodes a small non-JPEG at its own size", async () => {
  decoded = { width: 640, height: 480 };
  const out = await shrinkSelfie(new File(["png"], "small.png", { type: "image/png" }));
  expect(out.type).toBe("image/jpeg");
  expect(drawImage).toHaveBeenCalledWith(expect.anything(), 0, 0, 640, 480);
});

it("returns a small JPEG unchanged", async () => {
  decoded = { width: 1280, height: 720 };
  const file = new File(["jpg"], "small.jpg", { type: "image/jpeg" });
  expect(await shrinkSelfie(file)).toBe(file);
  expect(drawImage).not.toHaveBeenCalled();
});

it("returns the original file when it cannot be decoded", async () => {
  const file = new File(["junk"], "odd.webp", { type: "image/webp" });
  expect(await shrinkSelfie(file)).toBe(file);
});

it("returns the original file when the canvas cannot encode", async () => {
  decoded = { width: 4000, height: 3000 };
  vi.mocked(HTMLCanvasElement.prototype.toBlob).mockImplementation((cb) => cb(null));
  const file = new File(["big"], "big.jpg", { type: "image/jpeg" });
  expect(await shrinkSelfie(file)).toBe(file);
});
