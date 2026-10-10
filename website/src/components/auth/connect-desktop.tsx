"use client";

import { useState, type ReactNode } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { useAuth } from "@/hooks/use-auth";
import { ApiError } from "@/lib/api";
import { mintDesktopAuthCode } from "@/lib/api-account";
import { ROUTES } from "@/lib/constants";
import { buildDesktopCallbackUrl, isDesktopConnectParam, navigateTo } from "@/lib/desktop-connect";
import { cn } from "@/lib/utils";

// BatchMyPhotos opens /connect/desktop?state=…&challenge=…. The photographer
// signs in (ProtectedRoute bounces to /login and brings the query back),
// confirms, and the browser hands a one-time code to the app through a
// batchmyphotos:// link. No token ever travels in a URL.
// Contract: docs/desktop-upload-integration-guide.md § 1.
export function ConnectDesktop() {
  const params = useSearchParams();
  const state = params?.get("state");
  const challenge = params?.get("challenge");

  if (!isDesktopConnectParam(state) || !isDesktopConnectParam(challenge)) {
    return (
      <Copy heading="This connect link is not valid.">
        Start again from BatchMyPhotos: open the Upload view and press Connect.
      </Copy>
    );
  }

  return (
    <ProtectedRoute>
      <ConnectForm state={state} challenge={challenge} />
    </ProtectedRoute>
  );
}

type Phase =
  | { kind: "idle" }
  | { kind: "minting" }
  | { kind: "sent" }
  | { kind: "error"; message: string };

function ConnectForm({ state, challenge }: { state: string; challenge: string }) {
  const { user, logout } = useAuth();
  const [phase, setPhase] = useState<Phase>({ kind: "idle" });

  if (user && user.role !== "PHOTOGRAPHER") {
    return (
      <Copy heading="Connecting needs a photographer account.">
        You are signed in as {user.email}. Sign out, sign in with your photographer account, then press Connect in BatchMyPhotos again.
        <div className="pt-6">
          <button type="button" className={cn(BTN_SECONDARY, BTN_SIZE.md)} onClick={logout}>
            Sign out
          </button>
        </div>
      </Copy>
    );
  }

  async function connect() {
    setPhase({ kind: "minting" });
    try {
      const { code } = await mintDesktopAuthCode(challenge);
      navigateTo(buildDesktopCallbackUrl(code, state));
      setPhase({ kind: "sent" });
    } catch (error) {
      setPhase({ kind: "error", message: describeError(error) });
    }
  }

  if (phase.kind === "sent") {
    return (
      <Copy heading="Back to BatchMyPhotos.">
        The app should now show you as connected. If nothing happened, make sure BatchMyPhotos is open, then press Try again.
        <div className="flex flex-wrap gap-3 pt-6">
          <button type="button" className={cn(BTN_PRIMARY, BTN_SIZE.md)} onClick={connect}>
            Try again
          </button>
          <Link href={ROUTES.DASHBOARD} className={cn(BTN_GHOST, BTN_SIZE.md)}>
            Go to dashboard
          </Link>
        </div>
      </Copy>
    );
  }

  return (
    <Copy heading="Connect BatchMyPhotos to this account?">
      BatchMyPhotos will upload photos to your events as {user?.email}. You can disconnect from the app at any time.
      {phase.kind === "error" && (
        <p role="alert" className="pt-4 text-error">
          {phase.message}
        </p>
      )}
      <div className="flex flex-wrap gap-3 pt-6">
        <button
          type="button"
          className={cn(BTN_PRIMARY, BTN_SIZE.md)}
          onClick={connect}
          disabled={phase.kind === "minting"}
        >
          {phase.kind === "minting" ? "Connecting…" : "Connect"}
        </button>
        <Link href={ROUTES.DASHBOARD} className={cn(BTN_GHOST, BTN_SIZE.md)}>
          Not now
        </Link>
      </div>
    </Copy>
  );
}

function Copy({ heading, children }: { heading: string; children: ReactNode }) {
  return (
    <div className="space-y-5">
      <Kicker>BatchMyPhotos</Kicker>
      <h1 className="text-3xl md:text-4xl font-extrabold leading-tight">{heading}</h1>
      <div className="text-slate text-base leading-relaxed break-words">{children}</div>
    </div>
  );
}

function describeError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.errors.some((item) => item.code === "RATE_LIMITED")) {
      return "Too many attempts. Wait a minute and try again.";
    }
    return error.errors[0]?.message ?? "Could not create a connect code.";
  }
  return "Could not create a connect code. Check your connection and try again.";
}
