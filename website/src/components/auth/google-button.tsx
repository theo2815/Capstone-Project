"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/hooks/use-auth";
import { ROUTES } from "@/lib/constants";

interface GoogleButtonProps {
  disabled?: boolean;
}

export function GoogleButton({ disabled }: GoogleButtonProps) {
  const router = useRouter();
  const { mockGoogleLogin } = useAuth();
  const [isLoading, setIsLoading] = useState(false);

  async function handleClick() {
    if (isLoading || disabled) return;
    setIsLoading(true);
    try {
      await mockGoogleLogin();
      router.replace(ROUTES.ONBOARDING);
    } catch {
      setIsLoading(false);
    }
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={isLoading || disabled}
      className="w-full inline-flex items-center justify-center gap-3 bg-bone border border-line hover:border-slate hover:bg-bone-deep/40 active:bg-bone-deep py-3.5 rounded-full font-mono uppercase tracking-[0.2em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <GoogleG aria-hidden />
      <span>{isLoading ? "Connecting…" : "Continue with Google"}</span>
    </button>
  );
}

function GoogleG(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 18 18" className="size-4 shrink-0" {...props}>
      <path
        d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844a4.14 4.14 0 0 1-1.796 2.716v2.258h2.908c1.702-1.567 2.684-3.874 2.684-6.615z"
        fill="#4285F4"
      />
      <path
        d="M9 18c2.43 0 4.467-.806 5.956-2.184l-2.908-2.259c-.806.54-1.836.86-3.048.86-2.345 0-4.328-1.583-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z"
        fill="#34A853"
      />
      <path
        d="M3.964 10.706A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.706V4.962H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.038l3.007-2.332z"
        fill="#FBBC05"
      />
      <path
        d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.962L3.964 7.294C4.672 5.166 6.655 3.58 9 3.58z"
        fill="#EA4335"
      />
    </svg>
  );
}

export function AuthDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3" aria-hidden="true">
      <span className="flex-1 border-t border-line" />
      <span className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
        {label}
      </span>
      <span className="flex-1 border-t border-line" />
    </div>
  );
}
