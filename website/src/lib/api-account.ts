import { api } from "@/lib/api";
import type { User } from "@/types/user";

// Account backend contract (Phase F.1):
//   PUT /api/v1/me/profile  { name }                                → User
//   PUT /api/v1/me/password { currentPassword, newPassword }        → unknown (204 or { message })
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

export async function changePassword(args: ChangePasswordArgs): Promise<unknown> {
  return api.put<unknown>("/me/password", args);
}
