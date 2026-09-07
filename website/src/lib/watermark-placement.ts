import type { WatermarkPlacement } from "@/store/photographer-settings-store";

// Single source of truth for WHERE the photographer's logo lands, shared by the
// drag preview and mirrored exactly by the backend compositor
// (WatermarkService.drawWatermarkImage) so the preview is an accurate promise of
// the baked result. Free-space mapping: (0,0) pins the logo to the padded
// top-left, (1,1) to the padded bottom-right — so any value keeps it fully in
// frame and the same fraction lands consistently across photo sizes.

// Fraction of width used as the edge inset, both axes — matches the backend's
// `photo.width * 0.025`.
export const EDGE_PAD_RATIO = 0.025;

// Slider bounds — keep in sync with the backend clamps (PhotographerSettingsService).
export const OPACITY_MIN = 0.05;
export const OPACITY_MAX = 1;
export const SCALE_MIN = 0.05;
export const SCALE_MAX = 0.4;

export type Orientation = "landscape" | "portrait";

export interface WatermarkBox {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function posFor(p: WatermarkPlacement, o: Orientation): { x: number; y: number } {
  return o === "landscape" ? p.landscape : p.portrait;
}

// `logoAspect` = logo natural width / height; `boxW`/`boxH` = the preview
// element's pixel size. Returns the logo's CSS box inside it. Height follows the
// aspect ratio, so scaling never distorts the logo.
export function watermarkBox(
  boxW: number,
  boxH: number,
  logoAspect: number,
  scale: number,
  nx: number,
  ny: number,
): WatermarkBox {
  const pad = boxW * EDGE_PAD_RATIO;
  const width = boxW * scale;
  const height = logoAspect > 0 ? width / logoAspect : width;
  const freeW = Math.max(0, boxW - width - 2 * pad);
  const freeH = Math.max(0, boxH - height - 2 * pad);
  return {
    left: pad + clamp01(nx) * freeW,
    top: pad + clamp01(ny) * freeH,
    width,
    height,
  };
}

// Inverse of watermarkBox: given a dragged pixel top-left, the normalized (x,y)
// that reproduces it. Used by the drag handler to write back to the store.
export function positionFromBox(
  boxW: number,
  boxH: number,
  logoAspect: number,
  scale: number,
  left: number,
  top: number,
): { x: number; y: number } {
  const pad = boxW * EDGE_PAD_RATIO;
  const width = boxW * scale;
  const height = logoAspect > 0 ? width / logoAspect : width;
  const freeW = Math.max(1e-6, boxW - width - 2 * pad);
  const freeH = Math.max(1e-6, boxH - height - 2 * pad);
  return {
    x: clamp01((left - pad) / freeW),
    y: clamp01((top - pad) / freeH),
  };
}
