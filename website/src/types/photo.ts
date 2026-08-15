export interface Photo {
  id: string;
  eventId: string;
  thumbnailUrl: string;
  watermarkedUrl: string;
  tags: PhotoTag[];
  uploadedAt: string;
  photographerName: string;
}

export interface MockPhoto {
  id: string;
  bib: string | null;
  km: number | null;
  time: string;
  tone: number;
  span: "default" | "wide";
  price: number;
  imageUrl?: string | null;
  // Presigned URL for the clean original. Populated by the BE only when the
  // requester owns the photo (an unexpired DownloadGrant exists). When set,
  // the lightbox swaps `imageUrl` for `cleanUrl` so a runner who already
  // bought a photo sees an unwatermarked preview while browsing the event.
  // Null for everyone else — closes G-2.
  cleanUrl?: string | null;
  // Who took the shot, so a runner can tap through to /{handle}.
  // `photographerHandle` is null for a photographer who hasn't been verified
  // yet — the handle is only assigned at verification. A null handle means
  // "not linkable": render the name as plain text, never a link to /{null}.
  // Both are absent on legacy/seed rows that carry no photographer at all.
  photographerHandle?: string | null;
  photographerName?: string | null;
  alt?: string;
}

export interface PhotoTag {
  type: "FACE" | "BIB";
  value: string;
  confidence: number;
}

export interface SearchResult {
  photo: Photo;
  matchType: "FACE" | "BIB";
  confidence: number;
}

export interface FaceSearchRequest {
  eventId: string;
  image: File;
}

export interface BibSearchRequest {
  eventId: string;
  bibNumber: string;
}
