"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import Link from "next/link";
import { PlatformCutModal } from "@/components/dashboard/platform-cut-modal";
import { ROUTES } from "@/lib/constants";
import { formatPayoutNumber } from "@/lib/payout-format";
import { useScrollLock } from "@/lib/scroll-lock";
import {
  PAYOUT_METHOD_LABEL,
  usePhotographerSettingsStore,
} from "@/store/photographer-settings-store";

// Explainer that opens from the (How payouts work? →) trigger on
// /dashboard/billing. Same shape as PlatformCutModal: portals to body,
// scroll-locks, ESC + backdrop close, Quiet Studio styling.
//
// Sections: lede → four-step flow → where it lands (primary account
// preview) → processing time (per-method) → holds and withdrawals.
// The flow is photographer-initiated: earn → request (≥ ₱500) → admin
// reviews → admin transfers manually.

interface HowPayoutsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function HowPayoutsModal({ isOpen, onClose }: HowPayoutsModalProps) {
  const [mounted, setMounted] = useState(false);
  const [cutOpen, setCutOpen] = useState(false);
  const payouts = usePhotographerSettingsStore((s) => s.payouts);
  const primary = payouts.find((p) => p.isPrimary) ?? null;

  useEffect(() => {
    setMounted(true);
  }, []);

  useScrollLock(isOpen);

  // Pause this modal's ESC handler while the nested PlatformCutModal is open
  // so ESC closes only the top modal in the stack.
  useEffect(() => {
    if (!isOpen || cutOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEsc);
    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, cutOpen, onClose]);

  // Reset nested state whenever this modal closes so a re-open starts clean.
  useEffect(() => {
    if (!isOpen) setCutOpen(false);
  }, [isOpen]);

  if (!mounted || !isOpen) return null;

  const content = (
    <div
      className="fixed inset-0 z-[60] flex items-center justify-center p-4 md:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="how-payouts-title"
    >
      <div
        className="fixed inset-0 bg-ink/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      <div
        className="relative z-10 w-full max-w-xl max-h-[90vh] overflow-y-auto bg-bone border border-line rounded-lg shadow-xl"
        style={{ animation: "fade-up 0.25s ease-out both" }}
      >
        <div className="sticky top-0 z-10 flex items-baseline justify-between gap-6 px-6 md:px-8 pt-6 md:pt-8 pb-4 bg-bone border-b border-line">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
            Payouts
          </p>
          <button
            type="button"
            onClick={onClose}
            className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate hover:text-ink transition-colors"
            aria-label="Close"
          >
            Close ×
          </button>
        </div>

        <div className="px-6 md:px-8 pb-8 md:pb-10 pt-6 md:pt-8 space-y-10">
          <div>
            <h2
              id="how-payouts-title"
              className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink"
            >
              How payouts work
            </h2>
            <p className="font-sans text-sm md:text-base text-slate mt-3 max-w-md">
              You request payouts when ready. Once your unpaid earnings reach{" "}
              <span className="font-mono tnum">₱500</span>, hit{" "}
              <span className="text-ink">Request payout</span> — admin reviews
              and transfers to your primary payout account manually.
            </p>
            <button
              type="button"
              onClick={() => setCutOpen(true)}
              className="mt-4 inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors group focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone rounded-sm"
            >
              <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
                See how each sale is split
              </span>
              <span
                aria-hidden="true"
                className="transition-transform group-hover:translate-x-0.5"
              >
                →
              </span>
            </button>
          </div>

          <section aria-labelledby="flow-heading">
            <p
              id="flow-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              The flow
            </p>
            <FlowDiagram />
          </section>

          <section aria-labelledby="lands-heading">
            <p
              id="lands-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              Where it lands
            </p>
            <div className="mt-4 border border-line rounded-md p-5 md:p-6 bg-bone-deep/30">
              {primary ? (
                <>
                  <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
                    Primary account
                  </p>
                  <p className="font-display text-lg md:text-xl text-ink mt-2 tnum">
                    {PAYOUT_METHOD_LABEL[primary.method]}{" "}
                    <span className="text-slate-soft">·</span>{" "}
                    <span className="font-mono">
                      {formatPayoutNumber(
                        primary.method,
                        primary.accountNumber,
                      )}
                    </span>
                  </p>
                  <p className="font-sans text-sm text-slate mt-1 truncate">
                    {primary.accountName}
                  </p>
                </>
              ) : (
                <>
                  <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft">
                    No primary account yet
                  </p>
                  <p className="font-sans text-sm text-slate mt-2">
                    Add a payout account in settings to receive your earnings.
                  </p>
                </>
              )}
              <Link
                href={`${ROUTES.DASHBOARD_SETTINGS}#payouts`}
                onClick={onClose}
                className="mt-4 inline-flex items-center gap-1.5 font-sans text-sm text-slate hover:text-ink transition-colors group"
              >
                <span className="underline decoration-line underline-offset-4 decoration-1 group-hover:decoration-ink">
                  Manage payout accounts
                </span>
                <span
                  aria-hidden="true"
                  className="transition-transform group-hover:translate-x-0.5"
                >
                  →
                </span>
              </Link>
            </div>
            <p className="font-sans text-sm text-slate mt-4 max-w-md">
              Admin sends to this account when they approve your request — so
              keep the account number current before submitting.
            </p>
          </section>

          <section aria-labelledby="processing-heading">
            <p
              id="processing-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              Processing time
            </p>
            <p className="mt-3 font-sans text-sm text-slate max-w-md">
              Once admin transfers the funds, the bank-side processing time
              depends on the method:
            </p>
            <ul className="mt-4 border-y border-line divide-y divide-line">
              <ProcessingRow method="GCash" timing="instant" />
              <ProcessingRow method="Maya" timing="instant – 4 hours" />
              <ProcessingRow method="GoTyme" timing="4 – 24 hours" />
            </ul>
          </section>

          <section aria-labelledby="holds-heading">
            <p
              id="holds-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              Holds and withdrawals
            </p>
            <p className="mt-3 font-sans text-sm md:text-base text-slate max-w-md">
              If admin holds your request — usually because the payout account
              needs a fix — you&apos;ll see the reason on this page. Fix the
              issue, hit{" "}
              <span className="text-ink">Withdraw request</span>, then submit a
              new one. Refunded sales reduce your next available balance.
            </p>
          </section>
        </div>
      </div>
    </div>
  );

  return (
    <>
      {createPortal(content, document.body)}
      <PlatformCutModal
        isOpen={cutOpen}
        onClose={() => setCutOpen(false)}
      />
    </>
  );
}

function FlowDiagram() {
  return (
    <div className="mt-5">
      <div className="relative">
        <div
          aria-hidden="true"
          className="absolute top-1.5 left-[8%] right-[8%] h-px bg-line"
        />
        <div className="relative flex items-start justify-between gap-1">
          <FlowNode label="Earn" sub="From sales" />
          <FlowNode label="Request" sub="At ₱500+" />
          <FlowNode label="Review" sub="Admin OKs" />
          <FlowNode label="Paid" sub="Lands in account" active />
        </div>
      </div>
    </div>
  );
}

function FlowNode({
  label,
  sub,
  active = false,
}: {
  label: string;
  sub: string;
  active?: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-2 min-w-0 flex-1">
      <span
        aria-hidden="true"
        className={
          active
            ? "block w-3 h-3 rounded-full bg-fresh ring-4 ring-bone"
            : "block w-3 h-3 rounded-full bg-ink-soft ring-4 ring-bone"
        }
      />
      <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-ink tnum text-center">
        {label}
      </p>
      <p className="font-mono uppercase tracking-[0.2em] text-[10px] text-slate-soft text-center">
        {sub}
      </p>
    </div>
  );
}

function ProcessingRow({ method, timing }: { method: string; timing: string }) {
  return (
    <li className="py-3 md:py-4 flex items-baseline justify-between gap-4">
      <p className="font-sans text-sm md:text-base text-ink">{method}</p>
      <p className="font-mono tnum text-sm md:text-base text-slate">
        {timing}
      </p>
    </li>
  );
}
