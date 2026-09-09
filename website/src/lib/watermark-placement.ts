import type { WatermarkPlacement } from "@/store/photographer-settings-store";

// Single source of truth for WHERE the photographer's logo lands, shared by the
// drag preview and mirrored by WatermarkService.watermarkBox. All rounding and
// minimums apply in rendered-photo pixels, BEFORE scaling for the screen.
// (0,0) pins the fitted logo top-left; (1,1) pins it bottom-right.

// Fraction of width used as the edge inset, both axes — matches the backend's
// `photo.width * 0.025`.
export const EDGE_PAD_RATIO = 0.025;

// Slider bounds — keep in sync with the backend clamps (PhotographerSettingsService).
export const OPACITY_MIN = 0.05;
export const OPACITY_MAX = 1;
export const SCALE_MIN = 0.05;
export const SCALE_MAX = 0.4;

export type Orientation = "landscape" | "portrait";

// Representative 3:2 photos after the compositor's 1280px long-edge cap.
export const WATERMARK_PREVIEW_SIZE = {
  landscape: { width: 1280, height: 853 },
  portrait: { width: 853, height: 1280 },
} as const;

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

// `photoW`/`photoH` are the rendered photo dimensions, not CSS dimensions.
// Keep the backend's minimums and integer rounding, fitting both axes together
// when a tall logo or tiny photo would otherwise crop the mark.
export function watermarkBox(
  photoW: number,
  photoH: number,
  logoAspect: number,
  scale: number,
  nx: number,
  ny: number,
): WatermarkBox {
  const pad = Math.min(Math.max(12, Math.floor(photoW * EDGE_PAD_RATIO)), Math.floor((Math.min(photoW, photoH) - 1) / 2));
  const requestedWidth = Math.max(48, Math.floor(photoW * scale));
  const requestedHeight = Math.max(1, Math.floor(requestedWidth / logoAspect));
  const fit = Math.min(1, (photoW - 2 * pad) / requestedWidth, (photoH - 2 * pad) / requestedHeight);
  const width = Math.max(1, Math.floor(requestedWidth * fit));
  const height = Math.max(1, Math.floor(requestedHeight * fit));
  const freeW = photoW - width - 2 * pad;
  const freeH = photoH - height - 2 * pad;
  return {
    left: Math.round(pad + clamp01(nx) * freeW),
    top: Math.round(pad + clamp01(ny) * freeH),
    width,
    height,
  };
}

// A dragged top-left in rendered-photo pixels → normalized position. Reuse
// the fitted bounds so size limits and pixel rounding cannot drift on drag.
// Preserve an axis with no free space so shrinking the logo restores its position.
export function positionFromBox(
  photoW: number,
  photoH: number,
  logoAspect: number,
  scale: number,
  left: number,
  top: number,
  currentX: number,
  currentY: number,
): { x: number; y: number } {
  const start = watermarkBox(photoW, photoH, logoAspect, scale, 0, 0);
  const end = watermarkBox(photoW, photoH, logoAspect, scale, 1, 1);
  return {
    x: end.left > start.left ? clamp01((left - start.left) / (end.left - start.left)) : currentX,
    y: end.top > start.top ? clamp01((top - start.top) / (end.top - start.top)) : currentY,
  };
}
