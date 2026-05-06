"use client";

import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useScrollLock } from "@/lib/scroll-lock";

// Platform cut explainer. Triggered from the lifetime ₱ slab on
// /dashboard/earnings. Numbers are illustrative and based on a placeholder
// per-photo price of ₱125 — when per-photographer pricing lands in the
// settings store, swap PRICE / CUT to read from there (or from a backend
// config endpoint).
//
// TODO(backend): replace the hardcoded constants with a /api/platform/fees
// payload (cut, processing, breakdown). Until Phase F lands these values
// are duplicated here.

const PRICE = 125;
const CUT_RATE = 0.25;
const KEEP_RATE = 1 - CUT_RATE;
const CUT_AMOUNT = PRICE * CUT_RATE;
const KEEP_AMOUNT = PRICE * KEEP_RATE;
const SCALE_TIERS = [15, 30, 50, 100] as const;

interface PlatformCutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function PlatformCutModal({ isOpen, onClose }: PlatformCutModalProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useScrollLock(isOpen);

  useEffect(() => {
    if (!isOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEsc);
    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, onClose]);

  if (!mounted || !isOpen) return null;

  const content = (
    <div
      className="fixed inset-0 z-[70] flex items-center justify-center p-4 md:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="platform-cut-title"
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
            Platform cut
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
              id="platform-cut-title"
              className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink"
            >
              Where your peso goes
            </h2>
            <p className="font-sans text-sm md:text-base text-slate mt-3 max-w-md">
              QuickPitik keeps{" "}
              <span className="font-mono tnum text-ink">25%</span> of every
              photo sale. You keep the other{" "}
              <span className="font-mono tnum text-ink">75%</span>, paid out
              weekly.
            </p>
          </div>

          <section aria-labelledby="per-photo-heading">
            <p
              id="per-photo-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              One ₱{PRICE} photo
            </p>
            <ul className="mt-4 border-y border-line divide-y divide-line">
              <BreakdownRow
                label="Sale price"
                value={`₱${PRICE.toFixed(2)}`}
                tone="ink"
              />
              <BreakdownRow
                label="Platform cut (25%)"
                value={`−₱${CUT_AMOUNT.toFixed(2)}`}
                tone="slate"
              />
              <BreakdownRow
                label="You keep (75%)"
                value={`₱${KEEP_AMOUNT.toFixed(2)}`}
                tone="fresh"
                emphasized
              />
            </ul>
          </section>

          <section aria-labelledby="scale-heading">
            <p
              id="scale-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              At your pace
            </p>
            <ul className="mt-4 border-y border-line divide-y divide-line">
              {SCALE_TIERS.map((sold) => {
                const kept = sold * KEEP_AMOUNT;
                return (
                  <li
                    key={sold}
                    className="py-3 md:py-4 flex items-baseline justify-between gap-4"
                  >
                    <p className="font-sans text-sm md:text-base text-ink">
                      <span className="font-mono tnum text-ink">{sold}</span>{" "}
                      photos sold
                    </p>
                    <p className="font-mono tnum font-medium text-ink text-base md:text-lg">
                      ₱{kept.toLocaleString("en-PH", {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </p>
                  </li>
                );
              })}
            </ul>
            <p className="mt-3 font-mono uppercase tracking-[0.2em] text-[10px] text-slate-soft">
              Based on ₱{PRICE} per photo · weekly payout
            </p>
          </section>

          <section aria-labelledby="covers-heading">
            <p
              id="covers-heading"
              className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft"
            >
              What the 25% covers
            </p>
            <ul className="mt-4 space-y-3">
              <CoverRow>Photo hosting and bandwidth</CoverRow>
              <CoverRow>GCash, Maya, and GoTyme transfer fees</CoverRow>
              <CoverRow>AI face and bib search infrastructure</CoverRow>
              <CoverRow>Marketplace, downloads, and customer support</CoverRow>
            </ul>
          </section>
        </div>
      </div>
    </div>
  );

  return createPortal(content, document.body);
}

function BreakdownRow({
  label,
  value,
  tone,
  emphasized = false,
}: {
  label: string;
  value: string;
  tone: "ink" | "slate" | "fresh";
  emphasized?: boolean;
}) {
  const labelClass =
    tone === "fresh" ? "text-ink font-medium" : "text-slate";
  const valueClass =
    tone === "fresh"
      ? "text-fresh"
      : tone === "slate"
        ? "text-slate"
        : "text-ink";
  return (
    <li
      className={`flex items-baseline justify-between gap-4 ${emphasized ? "py-4 md:py-5" : "py-3 md:py-4"}`}
    >
      <p className={`font-sans text-sm md:text-base ${labelClass}`}>{label}</p>
      <p
        className={`font-mono tnum font-medium ${emphasized ? "text-xl md:text-2xl" : "text-base md:text-lg"} ${valueClass}`}
      >
        {value}
      </p>
    </li>
  );
}

function CoverRow({ children }: { children: React.ReactNode }) {
  return (
    <li className="flex items-baseline gap-3 font-sans text-sm md:text-base text-slate">
      <span aria-hidden="true" className="text-slate-soft mt-0.5">
        ·
      </span>
      <span>{children}</span>
    </li>
  );
}
