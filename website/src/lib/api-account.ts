import { api } from "@/lib/api";
import { getRefreshToken } from "@/lib/auth";
import type { User } from "@/types/user";

// Account backend contract (Phase F.1):
//   PUT /api/v1/me/profile  { name }                                              → User
//   PUT /api/v1/me/password { currentPassword, newPassword, refreshToken? }       → unknown (204 or { message })
//
// refreshToken is the local-storage refresh token of the current device. The BE
// hashes it and skips that one row when revoking — so password change kicks
// every OTHER session but keeps the user signed in here (plan C-7 / Phase E:
// "Server revokes all OTHER refresh tokens"). If localStorage was cleared
// between login and password change, the field is null and the BE falls back to
// revoking everything — the user gets logged out on this device too once the
// 15-min access token expires, which is the safe default.
//
// Error codes the FE handles inline:
//   profile:  VALIDATION_ERROR
//   password: INVALID_CREDENTIALS, VALIDATION_ERROR, SAME_PASSWORD

export async function updateProfileName(name: string): Promise<User> {
  return api.put<User>("/me/profile", { name });
}

export interface ChangePasswordArgs {
  currentPassword: string;
  newPassword: string;
}

export interface ChangePasswordResult {
  /**
   * False when localStorage had no refresh token to exempt, so the BE revoked
   * every session including this one — the user keeps a working access token
   * for ≤15 min and is then bounced. The caller must say so instead of
   * showing a plain success toast.
   */
  sessionKept: boolean;
}

export async function changePassword(
  args: ChangePasswordArgs,
): Promise<ChangePasswordResult> {
  const refreshToken = getRefreshToken();
  await api.put<unknown>("/me/password", { ...args, refreshToken });
  return { sessionKept: refreshToken !== null };
}
