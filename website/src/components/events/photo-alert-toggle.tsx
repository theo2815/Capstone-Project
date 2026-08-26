"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { usePathname, useRouter } from "next/navigation";
import { useAuthStore } from "@/store/auth-store";
import { useEffectiveRole } from "@/hooks/use-effective-role";
import { useSelfiesList } from "@/hooks/use-selfies";
import {
  fetchPhotoAlertStatus,
  registerPhotoAlert,
  unregisterPhotoAlert,
  type PhotoAlertStatus,
} from "@/lib/api-photo-alert";
import { ROUTES } from "@/lib/constants";
import { buildLoginRedirect } from "@/lib/redirect";
import { Kicker } from "@/components/ui/kicker";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

// "Get notified when your photos are ready" opt-in on the event page. Registers
// the runner's selfie for the backend sweep that emails them once when their
// matched photos appear. Runner-only; guests + selfie-less runners get a prompt
// (mirrors SelfieSearchPanel's auth/selfie affordances).
export function PhotoAlertToggle({ eventSlug }: { eventSlug: string }) {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const role = useEffectiveRole();
  const router = useRouter();
  const pathname = usePathname();
  const queryClient = useQueryClient();
  const { selfies, isLoading: selfiesLoading } = useSelfiesList();
  const here = pathname || `/events/${eventSlug}`;

  const hasSelfie = selfies.length > 0;
  const canToggle = isAuthenticated && hasSelfie;

  const statusQuery = useQuery<PhotoAlertStatus>({
    queryKey: ["photo-alert", eventSlug],
    queryFn: () => fetchPhotoAlertStatus(eventSlug),
    enabled: canToggle,
    staleTime: 60_000,
  });

  const registered = statusQuery.data?.registered ?? false;

  const mutation = useMutation({
    mutationFn: async (next: boolean) => {
      if (next) await registerPhotoAlert(eventSlug);
      else await unregisterPhotoAlert(eventSlug);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["photo-alert", eventSlug] });
    },
  });

  // Photographers don't receive race-photo alerts (no runner selfie library).
  if (role === "PHOTOGRAPHER") return null;

  const subtitle = !isAuthenticated
    ? "Sign in and we'll email you the moment we spot you."
    : selfiesLoading
      ? "We'll email you the moment your photos land."
      : !hasSelfie
        ? "Add a selfie and we'll email you the moment we spot you."
        : registered
          ? "You're on the list — we'll email you when your photos land."
          : "We'll email you the moment your photos land.";

  return (
    <div className="mt-4 rounded-2xl border border-line bg-bone-deep/60 px-5 py-4">
      <div className="flex items-start gap-3">
        <BellGlyph />
        <div className="flex-1 min-w-0">
          <p className="font-display font-bold text-ink text-base leading-snug">
            Get notified when your photos are ready
          </p>
          <p className="mt-0.5 font-sans text-sm text-slate leading-snug">
            {subtitle}
          </p>

          <div className="mt-3">
            {!isAuthenticated ? (
              <button
                type="button"
                onClick={() => router.push(buildLoginRedirect(here))}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                Sign in →
              </button>
            ) : selfiesLoading ? (
              <button
                type="button"
                disabled
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                Checking…
              </button>
            ) : !hasSelfie ? (
              <button
                type="button"
                onClick={() =>
                  router.push(
                    `${ROUTES.PROFILE}?next=${encodeURIComponent(here)}#selfies`,
                  )
                }
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                Add a selfie →
              </button>
            ) : registered ? (
              <div className="flex items-center gap-3">
                <Kicker
                  as="span"
                  className="inline-flex items-center gap-1.5 text-ink"
                >
                  <span
                    aria-hidden="true"
                    className="size-2 rounded-full bg-fresh"
                  />
                  Notifications on
                </Kicker>
                <button
                  type="button"
                  onClick={() => mutation.mutate(false)}
                  disabled={mutation.isPending}
                  className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-ink underline decoration-line underline-offset-4 disabled:opacity-50"
                >
                  Turn off
                </button>
              </div>
            ) : (
              <button
                type="button"
                onClick={() => mutation.mutate(true)}
                disabled={mutation.isPending || statusQuery.isPending}
                className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
              >
                {mutation.isPending ? "Turning on…" : "Notify me when ready"}
              </button>
            )}
          </div>

          {mutation.isError && (
            <p className="mt-2 font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-error">
              Couldn&apos;t update — try again.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function BellGlyph() {
  return (
    <span
      aria-hidden="true"
      className="mt-0.5 size-9 shrink-0 grid place-items-center rounded-full border border-line text-slate"
    >
      <svg
        viewBox="0 0 16 16"
        className="size-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M8 2.5a3.5 3.5 0 0 0-3.5 3.5c0 3-1.5 4-1.5 4h10s-1.5-1-1.5-4A3.5 3.5 0 0 0 8 2.5z" />
        <path d="M6.5 12.5a1.5 1.5 0 0 0 3 0" />
      </svg>
    </span>
  );
}
