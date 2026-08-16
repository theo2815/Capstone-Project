// Inline-validation helpers for /(auth)/* forms. Returns sentence-case error
// copy that ends with a period and names the constraint, per the FieldError
// tone convention. Returns null when the value is acceptable.
//
// Backend will perform its own server-side validation; these are the
// pre-submit hard gates that surface inline (no round-trip on obviously
// malformed input).

import { ApiError } from "@/lib/api";

// Shaped to agree with the backend's Jakarta `@Email`, which was rejecting
// addresses this gate waved through — `a@b..c`, `a@b.c.`, `a@-b.co` all cost a
// round-trip to learn "must be a well-formed email address".
//
//   local   dot-separated atoms of RFC-5322 atext — so no leading, trailing,
//           or consecutive dots
//   domain  labels that start and end alphanumeric (hyphens only inside)
//   TLD     2+ letters. Deliberately one notch stricter than Jakarta, which
//           accepts `a@b.c`; no single-letter TLD is deliverable.
//
// Quoted local parts (`"foo bar"@x.com`) are not supported. They are legal and
// effectively extinct, and the backend rejects them too.
const EMAIL_RE =
  /^[a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/i;

export const PASSWORD_MIN = 8;
export const NAME_MAX = 80;
export const EMAIL_MAX = 254;

// BCrypt hashes at most 72 bytes and silently discards the rest, so without a
// cap two visually different long passwords hash identically and both succeed
// at login. Measured in UTF-8 BYTES because that is the actual bcrypt limit —
// an ASCII password hits it at 72 characters, a multi-byte one sooner. The copy
// says "characters" because that is what a user counts; tripping early on
// multi-byte input is a safe failure, not a silent collision.
//
// Enforced by validateNewPassword() (register / reset / change) and NEVER by
// validatePassword() (login), which must go on accepting whatever an existing
// account was created with.
// Backend counterpart: `PasswordValidator`. Mobile: `ui/auth/AuthValidation.kt`.
export const PASSWORD_MAX_BYTES = 72;

export function validateEmail(value: string): string | null {
  const trimmed = value.trim();
  if (!trimmed) return "Email is required.";
  if (trimmed.length > EMAIL_MAX) return `Email is limited to ${EMAIL_MAX} characters.`;
  if (!EMAIL_RE.test(trimmed)) return "Use a valid email address.";
  return null;
}

/** Sign-in gate. Length floor only — never cap here, see PASSWORD_MAX_BYTES. */
export function validatePassword(value: string): string | null {
  if (!value) return "Password is required.";
  if (value.length < PASSWORD_MIN)
    return `Password must be at least ${PASSWORD_MIN} characters.`;
  return null;
}

/**
 * Gate for a password being SET — register, reset, change. Adds the bcrypt
 * ceiling on top of validatePassword()'s floor. Kept separate precisely so the
 * cap can never reach the login form.
 */
export function validateNewPassword(value: string): string | null {
  const floor = validatePassword(value);
  if (floor) return floor;
  if (new TextEncoder().encode(value).length > PASSWORD_MAX_BYTES)
    return `Password is limited to ${PASSWORD_MAX_BYTES} characters.`;
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
