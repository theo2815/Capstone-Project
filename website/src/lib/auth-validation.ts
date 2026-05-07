// Inline-validation helpers for /(auth)/* forms. Returns sentence-case error
// copy that ends with a period and names the constraint, per the FieldError
// tone convention. Returns null when the value is acceptable.
//
// Backend will perform its own server-side validation; these are the
// pre-submit hard gates that surface inline (no round-trip on obviously
// malformed input).

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
