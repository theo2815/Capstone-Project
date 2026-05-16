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
