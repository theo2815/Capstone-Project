// Client-side image processing for the user-media mock.
// Backend (Phase B+) will run the equivalent server-side at /me/avatar and
// /me/selfies upload — these helpers exist only so the website-only first
// pass produces images that resemble the eventual API payload (compressed,
// reasonably sized, JPEG).

const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export const ACCEPTED_IMAGE_MIME = Array.from(SUPPORTED_TYPES);
export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024; // 8 MB hard cap before processing

// Typed error so catch sites can produce specific copy instead of a single
// generic "Could not process this image." Codes:
//   decode         — image failed to load (corrupt / unsupported encoding)
//   canvas-tainted — getImageData blocked (CORS or cross-origin source)
//   encode         — canvas.toDataURL threw (out-of-memory or unsupported MIME)
//   memory         — explicit OOM signal (rare; iOS Safari hits this on huge files)
export type ImageProcessingCode =
  | "decode"
  | "canvas-tainted"
  | "encode"
  | "memory";

export class ImageProcessingError extends Error {
  code: ImageProcessingCode;
  constructor(message: string, code: ImageProcessingCode) {
    super(message);
    this.name = "ImageProcessingError";
    this.code = code;
  }
}

export function validateImageFile(file: File): string | null {
  if (!SUPPORTED_TYPES.has(file.type)) {
    return "Use a JPEG, PNG, or WebP image.";
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return "Image must be under 8 MB.";
  }
  return null;
}

export async function squareCropToDataUrl(
  file: File,
  size: number,
  quality = 0.85,
): Promise<string> {
  const img = await loadImage(file);
  const side = Math.min(img.width, img.height);
  const sx = (img.width - side) / 2;
  const sy = (img.height - side) / 2;

  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new ImageProcessingError(
      "Canvas 2D context unavailable.",
      "encode",
    );
  }
  ctx.drawImage(img, sx, sy, side, side, 0, 0, size, size);
  return safeToDataUrl(canvas, "image/jpeg", quality);
}

export async function fitToDataUrl(
  file: File,
  maxLongEdge: number,
  quality = 0.85,
): Promise<string> {
  const img = await loadImage(file);
  const { width, height } = scaleToFit(img.width, img.height, maxLongEdge);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new ImageProcessingError(
      "Canvas 2D context unavailable.",
      "encode",
    );
  }
  ctx.drawImage(img, 0, 0, width, height);
  return safeToDataUrl(canvas, "image/jpeg", quality);
}

// Same as fitToDataUrl but exports PNG so alpha transparency survives. Used
// for photographer watermark uploads where the watermark needs to overlay
// photos cleanly. Larger output than JPEG; scale watermark inputs accordingly.
export async function fitToPngDataUrl(
  file: File,
  maxLongEdge: number,
): Promise<string> {
  const img = await loadImage(file);
  const { width, height } = scaleToFit(img.width, img.height, maxLongEdge);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new ImageProcessingError(
      "Canvas 2D context unavailable.",
      "encode",
    );
  }
  ctx.drawImage(img, 0, 0, width, height);
  return safeToDataUrl(canvas, "image/png");
}

// Pre-flight inspector for slabs that need to react to dimensions before
// committing the upload (e.g., cover banner aspect-ratio check). Decodes the
// file once via the same private loadImage helper, so dimension reads share
// the same browser cache as the subsequent fitToDataUrl call.
export async function inspectImageDimensions(
  file: File,
): Promise<{ width: number; height: number; aspectRatio: number }> {
  const img = await loadImage(file);
  return {
    width: img.width,
    height: img.height,
    aspectRatio: img.width / img.height,
  };
}

// Heuristic transparency detector for watermark uploads. Downscales the
// source to 64×64 (the classification doesn't need full-res — solid-bg
// watermarks read clearly even at thumbnail size) and counts pixels with
// non-opaque alpha.
//
// alphaCoverage = fraction of pixels with alpha < 255
//   < 0.05 → effectively opaque (solid-bg PNG that will stamp as a block)
//   ≥ 0.05 → has meaningful transparency (overlay-friendly)
//
// hasAlpha = true if ANY pixel is non-opaque (cheap fast-path for catches that
// only care about the binary signal).
export async function detectPngTransparency(
  file: File,
): Promise<{ hasAlpha: boolean; alphaCoverage: number }> {
  const img = await loadImage(file);
  const sample = 64;
  const canvas = document.createElement("canvas");
  canvas.width = sample;
  canvas.height = sample;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new ImageProcessingError(
      "Canvas 2D context unavailable.",
      "encode",
    );
  }
  ctx.drawImage(img, 0, 0, sample, sample);
  let imageData: ImageData;
  try {
    imageData = ctx.getImageData(0, 0, sample, sample);
  } catch {
    throw new ImageProcessingError(
      "Browser blocked reading this image's pixels.",
      "canvas-tainted",
    );
  }
  const { data } = imageData;
  let translucent = 0;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] < 255) translucent++;
  }
  const total = sample * sample;
  return {
    hasAlpha: translucent > 0,
    alphaCoverage: translucent / total,
  };
}

function loadImage(file: File): Promise<HTMLImageElement> {
  const url = URL.createObjectURL(file);
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(
        new ImageProcessingError("Failed to load image.", "decode"),
      );
    };
    img.src = url;
  });
}

function safeToDataUrl(
  canvas: HTMLCanvasElement,
  type: string,
  quality?: number,
): string {
  try {
    return canvas.toDataURL(type, quality);
  } catch (err) {
    if (err instanceof Error && /tainted/i.test(err.message)) {
      throw new ImageProcessingError(
        "Browser blocked exporting this image.",
        "canvas-tainted",
      );
    }
    throw new ImageProcessingError(
      "Could not export the processed image.",
      "encode",
    );
  }
}

function scaleToFit(w: number, h: number, maxLongEdge: number) {
  if (w <= maxLongEdge && h <= maxLongEdge) return { width: w, height: h };
  const ratio = w >= h ? maxLongEdge / w : maxLongEdge / h;
  return { width: Math.round(w * ratio), height: Math.round(h * ratio) };
}
