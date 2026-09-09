import { api } from "@/lib/api";

// Admin TOTP MFA management (opt-in). Backend: /api/v1/admin/mfa/*.
// Enrollment is a two-step handshake — enroll returns the shared secret to add
// to an authenticator app, then confirm proves the app is set up and returns
// one-time recovery codes. Login enforcement is separate (AuthService).

export interface MfaStatus {
  enabled: boolean;
}

export interface MfaEnrollResponse {
  // otpauth:// URI for QR apps; `secret` is the same key for manual entry.
  otpauthUri: string;
  secret: string;
}

export interface MfaConfirmResponse {
  recoveryCodes: string[];
}

export async function fetchMfaStatus(): Promise<MfaStatus | null> {
  return api.get<MfaStatus>("/admin/mfa/status");
}

export async function enrollMfa(): Promise<MfaEnrollResponse> {
  return api.post<MfaEnrollResponse>("/admin/mfa/enroll");
}

export async function confirmMfa(code: string): Promise<MfaConfirmResponse> {
  return api.post<MfaConfirmResponse>("/admin/mfa/confirm", { code });
}

export async function disableMfa(code: string): Promise<void> {
  await api.post<void>("/admin/mfa/disable", { code });
}
