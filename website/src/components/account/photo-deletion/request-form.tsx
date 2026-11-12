"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import LiveSelfieCheck, { type LiveSelfieCheckProps, type LiveSelfieErrorKind } from "@/components/dashboard/live-selfie-check";
import { LegalText } from "@/components/legal/legal-document";
import { BTN_GHOST, BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Dropdown, DropdownItem } from "@/components/ui/dropdown";
import { Kicker } from "@/components/ui/kicker";
import { useAuth } from "@/hooks/use-auth";
import { useRunnerRemovalEligibility, useRunnerRemovalReasons } from "@/hooks/use-photo-removal";
import { usePublicEvents } from "@/hooks/use-public-events";
import { ApiError } from "@/lib/api";
import { fetchEventDetail } from "@/lib/api-events";
import { startRunnerLiveness, submitRunnerRemoval } from "@/lib/api-photo-removal";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { LEGAL_VERSION } from "@/lib/legal/version";
import { cn } from "@/lib/utils";
import type { RunnerRemovalBlockedReason } from "@/types/photo-removal";

const DETAILS_MAX = 500;
const LIST = ROUTES.ACCOUNT_PHOTO_DELETION;
const LIST_LINK = { href: LIST, label: "Your deletion requests →" };
const TEXT_LINK = "text-ink underline decoration-line-strong underline-offset-4 hover:decoration-fresh transition-colors";

interface PickedEvent { id: string; slug: string; name: string }
// What the page says after a refusal: the text, an optional link, and whether
// the live check may be started again (the email route always shows).
interface Notice { text: string; link?: { href: string; label: string }; again?: boolean; resend?: boolean }

const BLOCKED: Record<RunnerRemovalBlockedReason, string> = {
  EVENT_NOT_ELIGIBLE: "This event can't take deletion requests online.",
  AI_OFF: "Online deletion requests are switched off right now.",
  FACE_DATA_EXPIRED: "Face search has ended for this event, so we can't look for your photos online.",
  INTAKE_CLOSED: "This event no longer takes deletion requests online.",
  REQUEST_OPEN: "You already have an open request for this event.",
  TOO_MANY_OPEN: "You have as many open requests as we allow. Wait for one to close.",
  ATTEMPTS_EXHAUSTED: "You've used today's checks. Try again tomorrow or email us.",
};
const BLOCKED_EVENT = BLOCKED.EVENT_NOT_ELIGIBLE;
const BUSY = "Busy right now. Try again in a minute.";
const EMAIL_ONLY = "We can't take this request online right now.";

const STOPPED: Record<LiveSelfieErrorKind | "cancel", string> = {
  camera: "We couldn't use your camera. Allow camera access in your browser settings and try again.",
  load: "The live check didn't finish. It couldn't load.",
  connection: "The live check didn't finish. It lost its connection or ran out.",
  timeout: "The live check didn't finish in time.",
  other: "The live check didn't finish.",
  cancel: "The live check didn't finish. You closed it.",
};

function stoppedNotice(kind: LiveSelfieErrorKind | "cancel", triesLeft: number): Notice {
  const tries = triesLeft > 0 ? ` You have ${triesLeft} ${triesLeft === 1 ? "try" : "tries"} left today.` : "";
  return { text: STOPPED[kind] + tries, again: kind === "camera" || triesLeft > 0 };
}

// One refusal → one notice (RD-11 step 7). Unknown codes go to the email route.
function noticeFor(code: string, { resent, triesLeft }: { resent: boolean; triesLeft: number }): Notice {
  switch (code) {
    case "LIVENESS_UNAVAILABLE":
    case "RATE_LIMITED":
      return { text: BUSY, again: true };
    case "LIVENESS_FAILED":
      return { text: `We couldn't confirm a live check. You have ${triesLeft} ${triesLeft === 1 ? "try" : "tries"} left today.`, again: triesLeft > 0 };
    case "LIVENESS_ATTEMPTS_EXHAUSTED":
      return { text: BLOCKED.ATTEMPTS_EXHAUSTED };
    case "REMOVAL_REQUEST_OPEN":
      return { text: BLOCKED.REQUEST_OPEN, link: LIST_LINK };
    case "CONSENT_VERSION_MISMATCH":
      return { text: "Our privacy notice changed. Reload the page and tick again." };
    case "LIVENESS_SESSION_INVALID":
      return resent
        ? { text: "We may already have your request.", link: LIST_LINK }
        : { text: "That live check ran out. Take it again.", again: true };
    case "UPLOAD_STORAGE_UNAVAILABLE":
      return { text: "Your check could not be saved. Open your request to withdraw it and try again.", link: LIST_LINK };
    case "EVENT_NOT_FOUND":
      return { text: BLOCKED_EVENT };
    default: // AI_API_UNAVAILABLE, REMOVAL_INTAKE_CLOSED, FACE_SEARCH_ENDED and anything new
      return { text: EMAIL_ONLY };
  }
}

// A network drop or timeout: the same check can be sent again with the same key.
const canResend = (err: unknown) => !(err instanceof ApiError) || ["TIMEOUT", "UNKNOWN"].includes(err.errors[0]?.code ?? "");

export function EmailRoute() {
  return (
    <p className="font-sans text-sm leading-relaxed text-slate">
      <LegalText text={LEGAL_COPY.runnerDeletionEmailInstead} />
    </p>
  );
}

export function RequestForm({ eventSlug }: { eventSlug: string | null }) {
  const { user, logout } = useAuth();
  if (!user) return null;
  return (
    <div className="flex flex-col gap-8">
      <p className="font-sans text-sm text-slate">
        Signed in as <span className="font-semibold text-ink">{user.email}</span>. Not you?{" "}
        <button type="button" onClick={logout} className={TEXT_LINK}>Sign out</button>
      </p>
      {user.role === "RUNNER" ? (
        <RunnerFlow eventSlug={eventSlug} />
      ) : (
        <div className="flex flex-col gap-3 rounded-xl border border-line bg-bone-deep px-5 py-4">
          <p className="font-display text-lg font-bold text-ink">Runner accounts only</p>
          <p className="font-sans text-sm text-slate">Online deletion requests are for runner accounts.</p>
          <EmailRoute />
        </div>
      )}
    </div>
  );
}

function RunnerFlow({ eventSlug }: { eventSlug: string | null }) {
  const preset = useQuery({
    queryKey: ["public", "event-detail", eventSlug],
    queryFn: () => fetchEventDetail(eventSlug!),
    enabled: !!eventSlug,
  });
  const [picked, setPicked] = useState<PickedEvent | null>(null);
  const event: PickedEvent | null = picked ?? (preset.data ? { id: preset.data.id, slug: preset.data.slug, name: preset.data.name } : null);
  const eligibility = useRunnerRemovalEligibility(event?.id ?? null);

  return (
    <div className="flex flex-col gap-8">
      <section className="flex flex-col gap-3">
        <Kicker as="p">01 · Event</Kicker>
        {event ? (
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <p className="font-display text-xl font-bold text-ink">{event.name}</p>
            {picked && <button type="button" onClick={() => setPicked(null)} className={cn(BTN_GHOST, "text-sm")}>Change event</button>}
          </div>
        ) : (
          <EventSearch onPick={setPicked} />
        )}
      </section>
      {event && eligibility.data && (
        eligibility.data.eligible ? (
          <RequestSteps key={event.id} eventId={event.id} emailVerified={eligibility.data.emailVerified}
            attemptsLeftToday={eligibility.data.attemptsLeftToday} />
        ) : (
          <div className="flex flex-col gap-3 rounded-xl border border-line bg-bone-deep px-5 py-4">
            <p className="font-sans text-base text-ink">{BLOCKED[eligibility.data.blockedReason ?? "EVENT_NOT_ELIGIBLE"] ?? BLOCKED_EVENT}</p>
            {eligibility.data.blockedReason === "REQUEST_OPEN" && eligibility.data.openRequestId && (
              <Link href={`${LIST}/${eligibility.data.openRequestId}`} className={TEXT_LINK}>Open your request →</Link>
            )}
            <EmailRoute />
          </div>
        )
      )}
      {event && eligibility.isError && (
        <div className="flex flex-col gap-3">
          <p className="font-sans text-sm text-error">We couldn&apos;t check this event right now.</p>
          <EmailRoute />
        </div>
      )}
    </div>
  );
}

function EventSearch({ onPick }: { onPick: (event: PickedEvent) => void }) {
  const [term, setTerm] = useState("");
  const [search, setSearch] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setSearch(term.trim()), 300);
    return () => clearTimeout(t);
  }, [term]);
  // Without includeListed the catalog holds admin events only.
  const results = usePublicEvents({ search: search || undefined, includeListed: true, limit: 8 });
  return (
    <div className="flex flex-col gap-3">
      <label htmlFor="deletion-event" className="font-sans text-sm text-slate">Find your event</label>
      <input id="deletion-event" type="search" value={term} onChange={(e) => setTerm(e.target.value)} autoComplete="off"
        placeholder="Race name"
        className="w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors" />
      {term.trim() && results && (
        results.length ? (
          <ul className="flex flex-col divide-y divide-line rounded-xl border border-line bg-surface">
            {results.map((e) => (
              <li key={e.id}>
                <button type="button" onClick={() => onPick({ id: e.id, slug: e.slug, name: e.name })}
                  className="flex min-h-11 w-full items-center justify-between gap-3 px-4 py-3 text-left font-sans text-sm text-ink hover:bg-bone-deep/50">
                  <span className="truncate font-semibold">{e.name}</span>
                  <span className="shrink-0 font-mono text-sm text-slate-soft tnum">{e.date}</span>
                </button>
              </li>
            ))}
          </ul>
        ) : (
          <p className="font-sans text-sm text-slate">No events match. Check the race name.</p>
        )
      )}
    </div>
  );
}

type CheckProps = Omit<LiveSelfieCheckProps, "onError" | "onCancel"> & { stop: (kind: LiveSelfieErrorKind | "cancel") => void };
interface Pending { sessionId: string; key: string; triesLeft: number }

function RequestSteps({ eventId, emailVerified, attemptsLeftToday }: { eventId: string; emailVerified: boolean; attemptsLeftToday: number }) {
  const router = useRouter();
  const reasons = useRunnerRemovalReasons();
  const [reason, setReason] = useState<string | null>(null);
  const [details, setDetails] = useState("");
  const [ticked, setTicked] = useState(false);
  const [busy, setBusy] = useState<null | "starting" | "checking">(null);
  const [check, setCheck] = useState<CheckProps | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  // Kept after a dropped send so "Send again" reuses the session and its key.
  const pending = useRef<Pending | null>(null);
  const reasonLabel = reasons.data?.find((r) => r.code === reason)?.label;

  const submit = async (p: Pending, resent: boolean, isCurrent: () => boolean) => {
    pending.current = p;
    setNotice(null);
    setBusy("checking");
    try {
      const result = await submitRunnerRemoval(
        { sessionId: p.sessionId, eventId, reasonCode: reason!, details: details.trim() || undefined }, p.key);
      if (!isCurrent()) return;
      pending.current = null;
      router.push(`${LIST}/${encodeURIComponent(result.requestId)}`);
    } catch (err) {
      if (!isCurrent()) return;
      if (canResend(err)) {
        setNotice({ text: "We couldn't reach QuickPitik. Check your connection and send again.", resend: true });
      } else {
        pending.current = null;
        setNotice(noticeFor((err as ApiError).errors[0]?.code ?? "", { resent, triesLeft: p.triesLeft }));
      }
      setBusy(null);
    }
  };

  const start = async () => {
    if (busy || !reason || !ticked) return;
    const isCurrent = captureAuthSession();
    setNotice(null);
    setBusy("starting");
    try {
      const session = await startRunnerLiveness({ eventId, biometricConsent: true, consentVersion: LEGAL_VERSION });
      if (!isCurrent()) return;
      const { sessionId, region, credentials, attemptsLeftToday: triesLeft } = session;
      if (!credentials) { // only under Cognito, which AA-0.2 did not choose
        setNotice(stoppedNotice("load", triesLeft));
        setBusy(null);
        return;
      }
      setCheck({
        sessionId, region, credentials,
        onComplete: () => { setCheck(null); void submit({ sessionId, key: crypto.randomUUID(), triesLeft }, false, isCurrent); },
        stop: (kind) => { setCheck(null); setBusy(null); setNotice(stoppedNotice(kind, triesLeft)); },
      });
    } catch (err) {
      if (!isCurrent()) return;
      setBusy(null);
      setNotice(err instanceof ApiError ? noticeFor(err.errors[0]?.code ?? "", { resent: false, triesLeft: attemptsLeftToday }) : { text: BUSY, again: true });
    }
  };

  const resend = () => {
    if (pending.current && !busy) void submit(pending.current, true, captureAuthSession());
  };

  const stopped = notice && !notice.again && !notice.resend;
  return (
    <div className="flex flex-col gap-8">
      {!emailVerified && (
        <p className="rounded-xl border border-line bg-bone-deep px-5 py-4 font-sans text-sm leading-relaxed text-ink-soft">
          Verify your email so matching photos can be hidden at once. Until then our team reviews every photo.
        </p>
      )}

      <section className="flex flex-col gap-3">
        <Kicker as="p">02 · Reason</Kicker>
        <Dropdown ariaLabel="Reason" disabled={!!busy} trigger={
          <span className={cn("inline-flex w-full items-center justify-between gap-3 rounded-xl border border-line px-4 py-3 font-sans text-sm transition-colors hover:border-ink/40", reason ? "text-ink" : "text-slate")}>
            <span className="truncate">{reasonLabel ?? "Choose a reason…"}</span>
            <svg className="size-3 shrink-0 text-slate" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path d="M4 6 L8 10 L12 6" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </span>
        }>
          {(reasons.data ?? []).map((r) => (
            <DropdownItem key={r.code} onClick={() => setReason(r.code)} active={reason === r.code}>{r.label}</DropdownItem>
          ))}
        </Dropdown>
        <div className="flex flex-col gap-2">
          <label htmlFor="deletion-details" className="font-sans text-sm text-slate">Details (optional)</label>
          <textarea id="deletion-details" value={details} maxLength={DETAILS_MAX} rows={3} disabled={!!busy}
            onChange={(e) => setDetails(e.target.value)}
            className="w-full resize-y rounded-xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:border-fresh focus:outline-none" />
          <p className="self-end font-mono text-sm text-slate-soft tnum">{details.length}/{DETAILS_MAX}</p>
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <Kicker as="p">03 · Live check</Kicker>
        <div className="rounded-xl border border-line bg-bone-deep px-5 py-4 font-sans text-sm leading-relaxed text-ink-soft">
          <p><LegalText text={LEGAL_COPY.runnerDeletionNotice} newTab /></p>
          <label className="mt-3 flex min-h-11 cursor-pointer items-start gap-3 text-ink select-none">
            <input type="checkbox" checked={ticked} disabled={!!busy} onChange={(e) => setTicked(e.target.checked)}
              className="mt-1 size-4 shrink-0 accent-ink" />
            <span>{LEGAL_COPY.runnerDeletionTickLabel}</span>
          </label>
        </div>

        {notice && (
          <div className="flex flex-col gap-2" role="alert">
            <p className="font-sans text-sm text-error">{notice.text}</p>
            {notice.link && <Link href={notice.link.href} className={cn(TEXT_LINK, "w-fit text-sm")}>{notice.link.label}</Link>}
          </div>
        )}
        {busy === "checking" && <p role="status" className="font-sans text-sm text-slate">Checking your photos…</p>}

        {notice?.resend ? (
          <button type="button" onClick={resend} disabled={!!busy} className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-fit")}>Send again</button>
        ) : !stopped && (
          <button type="button" onClick={start} disabled={!reason || !ticked || !!busy} className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-fit")}>
            {busy === "starting" ? "Starting…" : "Start live check"}
          </button>
        )}
      </section>

      <EmailRoute />
      {check && <LiveSelfieCheck {...check} onError={check.stop} onCancel={() => check.stop("cancel")} />}
    </div>
  );
}
