"use client";

import { useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { Modal } from "@/components/ui/modal";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { putRacePrice, type RacePrice } from "@/lib/api-photographer";
import { captureAuthSession } from "@/lib/auth-session";
import { cn, formatPrice } from "@/lib/utils";

// AA-R16a: "my price for this race". Renders nothing unless the server sends
// racePrice (a race this photographer covers, self-serve races on). The range
// and the cooldown come from the server; it repeats every check.
export function RacePriceBand({ eventId, racePrice }: { eventId: string; racePrice?: RacePrice | null }) {
  const [open, setOpen] = useState(false);
  if (!racePrice) return null;
  return (
    <>
      <section className="mb-10 md:mb-12 border border-line rounded-2xl bg-bone-deep/30 p-6 md:p-8 flex flex-col md:flex-row md:items-center gap-4 md:gap-6">
        <div className="flex-1 min-w-0">
          <p className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            Your price
          </p>
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 tnum">
            {formatPrice(racePrice.effectivePhp)} a photo
          </p>
          <p className="font-sans text-sm text-ink-soft mt-2 tnum">
            {racePrice.myPricePhp === null ? "You charge the race price." : `The race price is ${formatPrice(racePrice.floorPhp)}.`}{" "}
            You can charge up to {formatPrice(racePrice.ceilingPhp)}.
          </p>
        </div>
        <div className="flex flex-wrap gap-2 shrink-0">
          <button type="button" onClick={() => setOpen(true)} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
            Change my price
          </button>
        </div>
      </section>
      {/* Mounted only while open, so each open starts from the server's price. */}
      {open && <RacePriceModal eventId={eventId} racePrice={racePrice} onClose={() => setOpen(false)} />}
    </>
  );
}

function RacePriceModal({ eventId, racePrice, onClose }: { eventId: string; racePrice: RacePrice; onClose: () => void }) {
  const queryClient = useQueryClient();
  const { showToast } = useToast();
  const [value, setValue] = useState(String(racePrice.effectivePhp));
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  const { floorPhp, ceilingPhp, nextChangeAt } = racePrice;
  const lockedUntil = nextChangeAt && new Date(nextChangeAt).getTime() > Date.now() ? new Date(nextChangeAt) : null;
  const amount = Number(value);
  const inRange = value.trim() !== "" && Number.isFinite(amount) && amount >= floorPhp && amount <= ceilingPhp;
  const disabled = saving || lockedUntil !== null;

  async function save(pricePhp: number | null) {
    if (saving) return;
    const current = captureAuthSession();
    setSaving(true);
    setError(null);
    try {
      await putRacePrice(eventId, pricePhp);
      if (!current()) return;
      await queryClient.invalidateQueries({ queryKey: ["photographer", "events"] });
      if (!current()) return;
      showToast({ kind: "success", message: "Your price is saved. Your photos in this race now use it." });
      onClose();
    } catch (err) {
      if (!current()) return;
      setError(err instanceof ApiError ? (err.errors[0]?.message ?? err.message) : "Couldn't save your price. Try again.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal isOpen onClose={onClose} title="My price for this race">
      <form
        className="space-y-5"
        onSubmit={(e) => {
          e.preventDefault();
          if (inRange && amount !== racePrice.effectivePhp) void save(amount);
        }}
      >
        <p className="font-sans text-sm text-ink-soft tnum">
          Charge from the race price {formatPrice(floorPhp)} up to {formatPrice(ceilingPhp)}. It applies to every photo you have in this race and to new uploads. After each change there is a wait before the next one.
        </p>
        <div className="space-y-2">
          <label htmlFor="race-price-input" className="block font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate">
            My price · PHP
          </label>
          <input
            id="race-price-input"
            type="number"
            inputMode="decimal"
            min={floorPhp}
            max={ceilingPhp}
            step="0.01"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            disabled={disabled}
            aria-invalid={value.trim() !== "" && !inRange}
            className="w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink tnum disabled:opacity-50 disabled:cursor-not-allowed"
          />
          {value.trim() !== "" && !inRange && (
            <p className="font-sans text-sm text-error tnum">Enter a price from {formatPrice(floorPhp)} to {formatPrice(ceilingPhp)}.</p>
          )}
        </div>
        {lockedUntil && (
          <p role="status" className="font-sans text-sm text-slate tnum">
            Next change after {lockedUntil.toLocaleString("en-PH", { dateStyle: "medium", timeStyle: "short" })}.
          </p>
        )}
        {error && <p role="alert" className="font-sans text-sm text-error">{error}</p>}
        <div className="flex flex-wrap gap-2">
          <button
            type="submit"
            disabled={disabled || !inRange || amount === racePrice.effectivePhp}
            className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
          >
            {saving ? "Saving…" : "Save price"}
          </button>
          <button
            type="button"
            onClick={() => void save(null)}
            disabled={disabled || racePrice.myPricePhp === null}
            className={cn(BTN_GHOST, BTN_SIZE.sm)}
          >
            Use the race price
          </button>
        </div>
      </form>
    </Modal>
  );
}
