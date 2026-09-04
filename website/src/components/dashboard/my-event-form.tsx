"use client";

import { type FormEvent, useEffect, useMemo, useState } from "react";
import { CoverField } from "@/components/admin/admin-event-form-modal";
import {
  AdminFieldLabel,
  AdminRadioGroup,
  AdminTextInput,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";
import {
  BTN_PRIMARY,
  BTN_SECONDARY,
  BTN_SIZE,
} from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { ApiError } from "@/lib/api";
import {
  createMyEvent,
  updateMyEvent,
  type MyEventFields,
  type PhotographerEventDetail,
} from "@/lib/api-photographer";
import { validateImageFile } from "@/lib/image-utils";
import { describePricing } from "@/lib/photographer-events";
import { cn } from "@/lib/utils";

// Photographer-owned event form (V46). Create = submit for admin review;
// edit = direct changes to title/date/location/notes/cover/visibility, while
// the pricing trio (paid/free · price · watermark) on a LIVE event becomes an
// edit request the admin approves — the form says so and never pretends the
// change is instant. The BE enforces the same rule; this only mirrors it.

const DATE_INPUT_CLS =
  "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh tnum";

type PricingMode = "paid" | "free";
type Watermark = "own" | "none";
type Visibility = "public" | "unlisted";

const PRICING_OPTIONS = [
  { value: "paid", label: "Paid — runners buy each photo, QuickPitik marks the previews" },
  {
    value: "free",
    label: "Free — anyone downloads the original. QuickPitik never marks free photos",
  },
] as const;

const WATERMARK_OPTIONS = [
  { value: "own", label: "My logo on the previews" },
  { value: "none", label: "No watermark at all" },
] as const;

const VISIBILITY_OPTIONS = [
  { value: "public", label: "Public — listed on /events for every runner" },
  { value: "unlisted", label: "Unlisted — only people with your link" },
] as const;

interface MyEventFormProps {
  /** Present in edit mode; the form prefills from it. */
  event?: PhotographerEventDetail;
  onDone: (saved: PhotographerEventDetail) => void;
  onCancel: () => void;
}

export function MyEventForm({ event, onDone, onCancel }: MyEventFormProps) {
  const live =
    event?.reviewStatus === "approved" ||
    event?.reviewStatus === "change_pending";
  // A parked request is what the photographer last asked for — prefill the
  // pricing section from it so a re-submit edits the request, not the live
  // settings.
  const requested = event?.pendingChange ?? null;

  const [title, setTitle] = useState(event?.name ?? "");
  const [location, setLocation] = useState(event?.location ?? "");
  const [date, setDate] = useState(event?.date ?? "");
  const [organizerName, setOrganizerName] = useState(
    event?.organizerName ?? "",
  );
  const [description, setDescription] = useState(event?.description ?? "");
  const [visibility, setVisibility] = useState<Visibility>(
    event?.visibility ?? "public",
  );
  const [pricingMode, setPricingMode] = useState<PricingMode>(
    requested?.pricingMode ?? event?.pricingMode ?? "paid",
  );
  const [price, setPrice] = useState(() => {
    const seed = requested
      ? Number(requested.pricePerPhoto)
      : (event?.pricePerPhoto ?? 0);
    return seed > 0 ? String(seed) : "";
  });
  const [watermark, setWatermark] = useState<Watermark>(
    (requested?.watermarkPolicy ?? event?.watermarkPolicy) === "none"
      ? "none"
      : "own",
  );
  const [cover, setCover] = useState<File | null>(null);
  const [coverError, setCoverError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const coverPreviewUrl = useMemo(
    () => (cover ? URL.createObjectURL(cover) : undefined),
    [cover],
  );
  useEffect(() => {
    return () => {
      if (coverPreviewUrl) URL.revokeObjectURL(coverPreviewUrl);
    };
  }, [coverPreviewUrl]);

  const parsedPrice = Number(price.trim());
  const priceValid =
    pricingMode === "free" ||
    (price.trim() !== "" && Number.isFinite(parsedPrice) && parsedPrice > 0);
  const dateValid = /^\d{4}-\d{2}-\d{2}$/.test(date);
  const canSubmit =
    title.trim().length > 0 &&
    location.trim().length > 0 &&
    dateValid &&
    priceValid &&
    !busy;

  const trio: MyEventFields =
    pricingMode === "free"
      ? { pricingMode: "free", pricePerPhoto: 0, watermarkPolicy: watermark }
      : {
          pricingMode: "paid",
          pricePerPhoto: parsedPrice,
          watermarkPolicy: "platform",
        };
  const trioChanged =
    !event ||
    event.pricingMode !== trio.pricingMode ||
    (event.watermarkPolicy ?? "platform") !== trio.watermarkPolicy ||
    Number(event.pricePerPhoto) !== trio.pricePerPhoto;

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    try {
      let saved: PhotographerEventDetail;
      if (!event) {
        saved = await createMyEvent({
          title: title.trim(),
          date,
          location: location.trim(),
          organizerName: organizerName.trim() || undefined,
          description: description.trim() || undefined,
          visibility,
          ...trio,
          cover,
        });
      } else {
        // Only what changed — blank is "no change" on the BE.
        const patch: MyEventFields = {};
        if (title.trim() !== event.name) patch.title = title.trim();
        if (date !== event.date) patch.date = date;
        if (location.trim() !== event.location)
          patch.location = location.trim();
        if (organizerName.trim() !== event.organizerName)
          patch.organizerName = organizerName.trim();
        if (description.trim() !== event.description)
          patch.description = description.trim();
        if (visibility !== event.visibility) patch.visibility = visibility;
        if (trioChanged) Object.assign(patch, trio);
        if (cover) patch.cover = cover;
        saved = await updateMyEvent(event.id, patch);
      }
      onDone(saved);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.errors[0]?.message ?? err.message)
          : "Couldn't save the event. Try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  const submitLabel = busy
    ? "Saving…"
    : !event
      ? "Submit for review"
      : live && trioChanged
        ? "Save · request pricing change"
        : "Save changes";

  return (
    <form onSubmit={handleSubmit} noValidate className="space-y-6">
      <AdminTextInput
        id="my-event-title"
        label="Title"
        value={title}
        onChange={setTitle}
        maxLength={200}
        showCounter
        placeholder="Paksit Sunrise Run"
        autoFocus={!event}
      />
      <AdminTextInput
        id="my-event-location"
        label="Location · Venue, City"
        value={location}
        onChange={setLocation}
        maxLength={200}
        showCounter
        placeholder="SRP Boulevard, Cebu City"
      />
      <div className="space-y-2">
        <AdminFieldLabel htmlFor="my-event-date">Date</AdminFieldLabel>
        <input
          id="my-event-date"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className={DATE_INPUT_CLS}
        />
      </div>
      <AdminTextInput
        id="my-event-organizer"
        label="Organizer · optional"
        value={organizerName}
        onChange={setOrganizerName}
        maxLength={120}
        showCounter
        placeholder="Your studio or the race organizer"
      />
      <AdminTextarea
        id="my-event-description"
        label="Race day notes · optional"
        value={description}
        onChange={setDescription}
        maxLength={600}
        rows={3}
        placeholder="Where you'll shoot, what runners should look for."
      />
      <CoverField
        bannerUrl={coverPreviewUrl ?? event?.bannerUrl ?? undefined}
        error={coverError}
        onPick={(file) => {
          const problem = validateImageFile(file);
          if (problem) {
            setCoverError(problem);
            return;
          }
          setCoverError(null);
          setCover(file);
        }}
        onRemove={() => {
          setCover(null);
          setCoverError(null);
        }}
      />

      <fieldset className="pt-5 border-t border-line space-y-4">
        <Kicker as="p">{live ? "Pricing · change request" : "Pricing"}</Kicker>
        {live && (
          <p className="font-sans text-sm text-ink-soft max-w-prose">
            Pricing on a live event changes only after an admin approves it.
            Until then the gallery keeps{" "}
            <span className="text-ink">
              {describePricing({
                pricingMode: event!.pricingMode ?? "paid",
                pricePerPhoto: event!.pricePerPhoto,
                watermarkPolicy: event!.watermarkPolicy ?? "platform",
              })}
            </span>
            .
          </p>
        )}
        {requested && (
          <p className="font-sans text-sm text-ink-soft">
            Waiting for approval:{" "}
            <span className="text-ink">{describePricing(requested)}</span>
          </p>
        )}
        <AdminRadioGroup
          name="my-event-pricing"
          options={PRICING_OPTIONS}
          value={pricingMode}
          onChange={(next) => setPricingMode(next as PricingMode)}
        />
        {pricingMode === "paid" ? (
          <AdminTextInput
            id="my-event-price"
            label="Price per photo · PHP"
            value={price}
            onChange={setPrice}
            type="number"
            min={1}
            step="0.01"
            inputMode="decimal"
            prefix="₱"
            placeholder="150"
            inputClassName="tnum"
            hint={
              price.trim() !== "" && !priceValid
                ? "Enter a price above zero (e.g. 150 or 99.50)."
                : undefined
            }
          />
        ) : (
          <div className="space-y-2">
            <AdminFieldLabel>Watermark on previews</AdminFieldLabel>
            <AdminRadioGroup
              name="my-event-watermark"
              options={WATERMARK_OPTIONS}
              value={watermark}
              onChange={(next) => setWatermark(next as Watermark)}
            />
          </div>
        )}
      </fieldset>

      <fieldset className="pt-5 border-t border-line space-y-4">
        <Kicker as="p">Visibility</Kicker>
        <AdminRadioGroup
          name="my-event-visibility"
          options={VISIBILITY_OPTIONS}
          value={visibility}
          onChange={(next) => setVisibility(next as Visibility)}
        />
      </fieldset>

      {!event && (
        <p className="font-sans text-sm text-ink-soft max-w-prose">
          An admin reviews the event and its price first. Uploads open the
          moment it is approved.
        </p>
      )}
      {error && (
        <p role="alert" className="font-sans text-sm text-error">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-3 pt-2">
        <button
          type="submit"
          disabled={!canSubmit}
          className={cn(BTN_PRIMARY, BTN_SIZE.sm)}
        >
          {submitLabel}
        </button>
        <button
          type="button"
          onClick={onCancel}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Cancel
        </button>
      </div>
    </form>
  );
}
