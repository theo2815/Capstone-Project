"use client";

import { type FormEvent, useEffect, useId, useMemo, useState } from "react";
import {
  AdminFieldLabel,
  AdminTextInput,
} from "@/components/admin/admin-form-fields";
import {
  BTN_DANGER,
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { Modal } from "@/components/ui/modal";
import { usePlatformFees } from "@/hooks/use-photographer-data";
import { ApiError } from "@/lib/api";
import {
  deleteEventCoupon,
  fetchEventCoupon,
  putEventCoupon,
} from "@/lib/api-coupons";
import type { PhotographerEventSummary } from "@/lib/photographer-mock";
import { cn } from "@/lib/utils";

type CouponEvent = Pick<
  PhotographerEventSummary,
  "id" | "name" | "ownedByMe" | "pricingMode"
>;

interface EventCouponModalProps {
  isOpen: boolean;
  onClose: () => void;
  events: readonly CouponEvent[];
  initialEventId?: string;
  lockEvent?: boolean;
}

const INPUT_CLS =
  "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh";
const MAX_USAGE_LIMIT = 1_000_000;

export function EventCouponModal({
  isOpen,
  onClose,
  events,
  initialEventId,
  lockEvent = false,
}: EventCouponModalProps) {
  const eligibleEvents = useMemo(
    () => events.filter((event) => event.ownedByMe && event.pricingMode === "paid"),
    [events],
  );
  const initial = eligibleEvents.some((event) => event.id === initialEventId)
    ? initialEventId!
    : (eligibleEvents[0]?.id ?? "");
  const [eventId, setEventId] = useState(initial);
  const [code, setCode] = useState("");
  const [percentOff, setPercentOff] = useState("10");
  const [expiresOn, setExpiresOn] = useState("");
  const [usageLimit, setUsageLimit] = useState("");
  const [active, setActive] = useState(true);
  const [exists, setExists] = useState(false);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const id = useId();
  const fees = usePlatformFees();

  useEffect(() => {
    if (isOpen) setEventId(initial);
  }, [initial, isOpen]);

  useEffect(() => {
    if (!isOpen || !eventId) return;
    let current = true;
    setLoading(true);
    setError(null);
    void fetchEventCoupon(eventId)
      .then((coupon) => {
        if (!current) return;
        setExists(coupon !== null);
        setCode(coupon?.code ?? "");
        setPercentOff(String(coupon?.percentOff ?? 10));
        setExpiresOn(coupon?.expiresAt?.slice(0, 10) ?? "");
        setUsageLimit(coupon?.usageLimit ? String(coupon.usageLimit) : "");
        setActive(coupon?.active ?? true);
      })
      .catch((err) => {
        if (current) setError(messageOf(err, "Couldn't load this coupon."));
      })
      .finally(() => {
        if (current) setLoading(false);
      });
    return () => {
      current = false;
    };
  }, [eventId, isOpen]);

  const percent = Number(percentOff);
  const limit = usageLimit === "" ? null : Number(usageLimit);
  const valid =
    /^[A-Z0-9]{4,16}$/.test(code) &&
    Number.isInteger(percent) &&
    percent >= 1 &&
    percent <= fees.couponMaxPercent &&
    (limit === null ||
      (Number.isInteger(limit) && limit > 0 && limit <= MAX_USAGE_LIMIT));

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!eventId || !valid) return;
    setBusy(true);
    setError(null);
    try {
      await putEventCoupon(eventId, {
        code,
        percentOff: percent,
        active,
        expiresAt: expiresOn
          ? new Date(`${expiresOn}T23:59:59`).toISOString()
          : null,
        usageLimit: limit,
      });
      onClose();
    } catch (err) {
      setError(messageOf(err, "Couldn't save the coupon."));
    } finally {
      setBusy(false);
    }
  }

  async function handleDelete() {
    if (!eventId) return;
    setBusy(true);
    setError(null);
    try {
      await deleteEventCoupon(eventId);
      onClose();
    } catch (err) {
      setError(messageOf(err, "Couldn't delete the coupon."));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Event coupon">
      {eligibleEvents.length === 0 ? (
        <p className="font-sans text-sm text-ink-soft">
          Create a paid event before adding a coupon.
        </p>
      ) : (
        <form onSubmit={handleSubmit} noValidate className="space-y-6">
          <p className="font-sans text-sm text-ink-soft">
            This coupon automatically covers every paid photo in the selected
            event that belongs to you.
          </p>

          <div className="space-y-2">
            <AdminFieldLabel htmlFor={`${id}-event`}>Event</AdminFieldLabel>
            <select
              id={`${id}-event`}
              value={eventId}
              onChange={(event) => setEventId(event.target.value)}
              disabled={lockEvent || busy}
              className={INPUT_CLS}
            >
              {eligibleEvents.map((event) => (
                <option key={event.id} value={event.id}>
                  {event.name}
                </option>
              ))}
            </select>
          </div>

          <AdminTextInput
            id={`${id}-code`}
            label="Coupon code"
            value={code}
            onChange={(value) => setCode(value.toUpperCase())}
            sanitize={(value) => value.replace(/[^A-Za-z0-9]/g, "")}
            maxLength={16}
            placeholder="RACE10"
            autoFocus
            hint={code.length > 0 && !/^[A-Z0-9]{4,16}$/.test(code) ? "Use 4–16 letters or numbers." : undefined}
          />
          <AdminTextInput
            id={`${id}-percent`}
            label={`Discount · 1–${fees.couponMaxPercent}%`}
            type="number"
            inputMode="numeric"
            min={1}
            max={fees.couponMaxPercent}
            value={percentOff}
            onChange={setPercentOff}
            inputClassName="tnum"
          />

          <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
            <div className="space-y-2">
              <AdminFieldLabel htmlFor={`${id}-expiry`}>
                Expiry · optional
              </AdminFieldLabel>
              <input
                id={`${id}-expiry`}
                type="date"
                min={new Date().toISOString().slice(0, 10)}
                value={expiresOn}
                onChange={(event) => setExpiresOn(event.target.value)}
                className={cn(INPUT_CLS, "tnum")}
              />
            </div>
            <AdminTextInput
              id={`${id}-limit`}
              label="Uses · optional"
              type="number"
              inputMode="numeric"
              min={1}
              max={MAX_USAGE_LIMIT}
              value={usageLimit}
              onChange={setUsageLimit}
              placeholder="Unlimited"
              inputClassName="tnum"
            />
          </div>

          <label className="flex items-center gap-3 font-sans text-sm text-ink-soft">
            <input
              type="checkbox"
              checked={active}
              onChange={(event) => setActive(event.target.checked)}
              className="size-4 accent-ink"
            />
            Active now
          </label>

          {error && (
            <p role="alert" className="font-sans text-sm text-error">
              {error}
            </p>
          )}

          <div className="flex flex-wrap justify-end gap-2">
            {exists && (
              <button
                type="button"
                onClick={handleDelete}
                disabled={busy}
                className={cn(BTN_DANGER, BTN_SIZE.sm, "mr-auto")}
              >
                Delete
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              disabled={busy}
              className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!valid || loading || busy}
              className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
            >
              {busy ? "Saving…" : exists ? "Save coupon" : "Create coupon"}
            </button>
          </div>
        </form>
      )}
    </Modal>
  );
}

function messageOf(error: unknown, fallback: string): string {
  return error instanceof ApiError
    ? (error.errors[0]?.message ?? error.message)
    : fallback;
}
