import { api } from "@/lib/api";
import type { User } from "@/types/user";

// Avatar backend contract (Q-007 RESOLVED 2026-05-09):
//   POST   /api/v1/me/avatar  (multipart, field "file")  → User (avatarUrl populated)
//   DELETE /api/v1/me/avatar                              → User (avatarUrl cleared)
//
// Server mirrors the client-side 512×512 center-crop + JPEG re-encode that
// `lib/image-utils.ts:squareCropToDataUrl` runs. Returning the full User lets
// the FE update auth-store atomically via setUser(response).

export async function uploadAvatar(file: File): Promise<User> {
  const formData = new FormData();
  formData.append("file", file);
  return api.post<User>("/me/avatar", formData);
}

export async function deleteAvatar(): Promise<User> {
  return api.delete<User>("/me/avatar");
}
