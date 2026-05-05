import { useSearchParams } from "next/navigation";
import { ROUTES } from "@/lib/constants";

export function isSafeRedirect(value: string | null | undefined): value is string {
  if (!value) return false;
  if (!value.startsWith("/")) return false;
  if (value.startsWith("//") || value.startsWith("/\\")) return false;
  return true;
}

export function useRedirectTarget(fallback: string = ROUTES.HOME): string {
  const params = useSearchParams();
  const raw = params?.get("redirect");
  return isSafeRedirect(raw) ? raw : fallback;
}
