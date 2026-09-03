import { api } from "@/lib/api";

// Mirrors backend PhotoVerifyResultDto. Attribution only — the backend never
// returns a photo id or URL for a verify hit, by design.
export interface PhotoVerifyResult {
  matched: boolean;
  confidence: "strong" | "weak" | null;
  photographerName: string | null;
  photographerHandle: string | null;
  eventName: string | null;
  /** ISO date (yyyy-mm-dd). */
  eventDate: string | null;
}

// POST /public/photos/verify — public, IP rate-limited (10 / 15 min). A 429
// arrives as ApiError with retryAfterSeconds.
export async function verifyPhoto(file: File): Promise<PhotoVerifyResult> {
  const body = new FormData();
  body.append("file", file, file.name);
  return api.post<PhotoVerifyResult>("/public/photos/verify", body);
}
