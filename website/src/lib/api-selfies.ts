import { api } from "@/lib/api";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import { useUserMediaStore, type SelfieRef } from "@/store/user-media-store";

// Selfies backend contract (Q-006 RESOLVED 2026-05-09):
//   GET    /api/v1/me/selfies                         → SelfieRef[] (max 5)
//   POST   /api/v1/me/selfies          (multipart)    → SelfieRef
//   DELETE /api/v1/me/selfies/{id}                    → { removed: boolean }
//   POST   /api/v1/me/selfies/{id}/set-primary        → SelfieRef[] (canonical list, new primary set)
//
// Errors handled inline:
//   SELFIE_REJECTED         — 422 with verbatim ai-api reason ("no face", "too blurry", "multiple faces")
//   SELFIE_LIMIT_REACHED    — 409 cap=5 reached
//
// FE keeps the field name `dataUrl` for now (string is signed S3 URL in live mode,
// base64 data URL in mock mode). Rename to `imageUrl` deferred to a follow-up commit.

export async function fetchSelfies(): Promise<SelfieRef[]> {
  if (BACKEND_LIVE) {
    return api.get<SelfieRef[]>("/me/selfies");
  }
  return useUserMediaStore.getState().selfies;
}

export async function uploadSelfie(file: File): Promise<SelfieRef> {
  const formData = new FormData();
  formData.append("file", file);
  return api.post<SelfieRef>("/me/selfies", formData);
}

export async function deleteSelfie(id: string): Promise<{ removed: boolean }> {
  return api.delete<{ removed: boolean }>(
    `/me/selfies/${encodeURIComponent(id)}`,
  );
}

export async function setPrimarySelfie(id: string): Promise<SelfieRef[]> {
  return api.post<SelfieRef[]>(
    `/me/selfies/${encodeURIComponent(id)}/set-primary`,
  );
}
