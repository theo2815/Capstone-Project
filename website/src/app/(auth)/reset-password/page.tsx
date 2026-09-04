import { redirect } from "next/navigation";
import { ROUTES } from "@/lib/constants";

// The link-token reset retired with the OTP cutover — the whole flow now
// lives on /forgot-password. Kept as a redirect for stale bookmarks.
export default function ResetPasswordPage() {
  redirect(ROUTES.FORGOT_PASSWORD);
}
