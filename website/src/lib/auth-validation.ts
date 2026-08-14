// Inline-validation helpers for /(auth)/* forms. Returns sentence-case error
// copy that ends with a period and names the constraint, per the FieldError
// tone convention. Returns null when the value is acceptable.
//
// Backend will perform its own server-side validation; these are the
// pre-submit hard gates that surface inline (no round-trip on obviously
// malformed input).

import { ApiError } from "@/lib/api";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export const PASSWORD_MIN = 8;
export const NAME_MAX = 80;
export const EMAIL_MAX = 254;

export function validateEmail(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return "Email is required.";
  if (trimmed.length > EMAIL_MAX) return `Email is limited to ${EMAIL_MAX} characters.`;
  if (!EMAIL_RE.test(trimmed)) return "Use a valid email address.";
  return null;
}

export function validatePassword(value: string): string | null {
  if (!value) return "Password is required.";
  if (value.length < PASSWORD_MIN)
    return `Password must be at least ${PASSWORD_MIN} characters.`;
  return null;
}

export function validateName(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return "Name is required.";
  if (trimmed.length > NAME_MAX)
    return `Name is limited to ${NAME_MAX} characters.`;
  return null;
}

// Split a thrown ApiError into per-field messages plus a residual top-level
// message, so server-side validation lands under the input it's about instead
// of collapsing into one banner above the submit button.
//
// The backend already ships the data: `GlobalExceptionHandler.handleMethodArgNotValid`
// maps each Bean Validation failure to `ApiError(field = it.field)`.
//
// `renderable` maps a backend field name onto the local state key that has an
// input to hang it under. It is an ALLOW-LIST, not just a rename table — a
// backend field with no entry has no place to render (reset's `token`, or a
// `role` failure on register), so its message falls through to the top-level
// slot rather than being written to a key nothing displays. Field-less errors
// (INVALID_CREDENTIALS, ACCOUNT_SUSPENDED) take the same path. Nothing is
// dropped except a second message for a field that already has one.
export function splitApiFieldErrors(
  err: unknown,
  renderable: Record<string, string>,
): { fields: Record<string, string>; message: string | null } {
  const fields: Record<string, string> = {};
  const rest: string[] = [];

  if (!(err instanceof ApiError)) return { fields, message: null };

  for (const e of err.errors) {
    const key = e.field ? renderable[e.field] : undefined;
    if (!key) {
      rest.push(e.message);
    } else if (!(key in fields)) {
      // First message per field wins — a second failure on the same input
      // would otherwise replace the one the user is already reading.
      fields[key] = e.message;
    }
  }

  return { fields, message: rest[0] ?? null };
}
