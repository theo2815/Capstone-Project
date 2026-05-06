// Client-side image processing for the user-media mock.
// Backend (Phase B+) will run the equivalent server-side at /me/avatar and
// /me/selfies upload — these helpers exist only so the website-only first
// pass produces images that resemble the eventual API payload (compressed,
// reasonably sized, JPEG).

const SUPPORTED_TYPES = new Set(["image/jpeg", "image/png", "image/webp"]);

export const ACCEPTED_IMAGE_MIME = Array.from(SUPPORTED_TYPES);
export const MAX_UPLOAD_BYTES = 8 * 1024 * 1024; // 8 MB hard cap before processing

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
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(img, sx, sy, side, side, 0, 0, size, size);
  return canvas.toDataURL("image/jpeg", quality);
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
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(img, 0, 0, width, height);
  return canvas.toDataURL("image/jpeg", quality);
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
  if (!ctx) throw new Error("Canvas 2D context unavailable.");
  ctx.drawImage(img, 0, 0, width, height);
  return canvas.toDataURL("image/png");
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
      reject(new Error("Failed to load image."));
    };
    img.src = url;
  });
}

function scaleToFit(w: number, h: number, maxLongEdge: number) {
  if (w <= maxLongEdge && h <= maxLongEdge) return { width: w, height: h };
  const ratio = w >= h ? maxLongEdge / w : maxLongEdge / h;
  return { width: Math.round(w * ratio), height: Math.round(h * ratio) };
}
