import type { OrderPhotoDetail } from "@/types/order";

// Per-photo download helpers shared by /orders/return (PhotoCard) and /orders
// (lightbox owned-mode CTA).
//
// Why both pages need this: `<a download>` is silently ignored when the URL
// is cross-origin — true for any S3 presigned URL and true for the LocalFs
// dev backend (served from the BE on :8080). Setting
//   Content-Disposition: attachment; filename="…"
// on the response forces the browser to save instead of navigate, regardless
// of origin or whether `<a download>` is honored.
//
// LocalFs path: `StorageDownloadDispositionFilter` on the BE reads the
// `disposition` + `filename` query params off `/storage/*` URLs and writes
// the header. S3 has native equivalents (`response-content-disposition`) —
// not appended here yet because dev runs LocalFs; revisit when prod S3
// flips on.

export function buildPhotoDownloadFilename(photo: OrderPhotoDetail): string {
  const tag = photo.bib
    ? `bib-${photo.bib}`
    : `untagged-${photo.id.slice(0, 8)}`;
  return `quickpitik-${tag}.jpg`;
}

export function appendDownloadDisposition(
  url: string,
  filename: string,
): string {
  const sep = url.includes("?") ? "&" : "?";
  const encodedName = encodeURIComponent(filename);
  return `${url}${sep}disposition=attachment&filename=${encodedName}`;
}
