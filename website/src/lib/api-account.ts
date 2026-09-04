import { api } from "@/lib/api";
import { getRefreshToken } from "@/lib/auth";
import type { User } from "@/types/user";

// Account backend contract (Phase F.1):
//   PUT /api/v1/me/profile  { name }                                              → User
//   PUT /api/v1/me/password { currentPassword, newPassword, refreshToken? }       → unknown (204 or { message })
//   PUT  /api/v1/me/email               { newEmail, currentPassword }             → { message }
//   POST /api/v1/auth/confirm-email-change { token }   (PUBLIC)                   → { message }
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
//   email:    INVALID_CREDENTIALS, SAME_EMAIL, EMAIL_TAKEN, VALIDATION_ERROR
//   confirm:  INVALID_EMAIL_CHANGE_TOKEN, EMAIL_TAKEN

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

export interface RequestEmailChangeArgs {
  newEmail: string;
  currentPassword: string;
}

/**
 * Step 1 of 2. Does NOT change the sign-in email — it mails a confirmation
 * link to the NEW address, and nothing moves until that link is redeemed from
 * that inbox. Callers must not report success as "email updated"; the returned
 * `message` is the backend's own wording and says what actually happened.
 */
export async function requestEmailChange(
  args: RequestEmailChangeArgs,
): Promise<string> {
  const res = await api.put<{ message?: string }>("/me/email", args);
  return (
    res?.message ??
    "Check your new inbox for the confirmation link. Your sign-in email stays the same until you use it."
  );
}

/**
 * Step 2 of 2. Public — the link opens from the new inbox, which is often a
 * browser with no session, so the opaque token is the only credential. On
 * success the backend revokes every refresh token, so any local session is
 * already dead and the caller must clear it.
 */
export async function confirmEmailChange(token: string): Promise<void> {
  await api.post<unknown>("/auth/confirm-email-change", { token });
}
