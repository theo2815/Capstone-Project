"use client";

import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFieldHint,
  AdminFieldLabel,
  AdminTextInput,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";
import type { ListEvent } from "@/app/events/events-browser";
import {
  ACCEPTED_IMAGE_MIME,
  validateImageFile,
} from "@/lib/image-utils";
import { cn } from "@/lib/utils";

// Create / edit form modal for admin events. Fields: Title, Location, Date,
// and an optional cover banner. The cover File is handed back to the caller
// raw — the backend (EventCoverService) center-crops to 4:3 and re-encodes
// to JPEG. Sending the bytes through multipart avoids the data-URL detour
// that overflowed banner_url VARCHAR(512) and 500'd the create.
//
// Mode prop selects copy + button label. In `edit` mode the form prefills
// from the passed `event`; the date input enforces ISO YYYY-MM-DD. Cover
// edits over PATCH /admin/events/{id} aren't wired yet — picking a file in
// edit mode just stages it locally.

const DATE_INPUT_CLS =
  "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh tnum";

type Mode = "create" | "edit";

interface AdminEventFormModalProps {
  open: boolean;
  mode: Mode;
  event?: ListEvent | null;
  onClose: () => void;
  onSubmit: (payload: {
    name: string;
    location: string;
    date: string;
    /** Per-photo price in PHP (admin-set). On edit, send only when changed
     *  from the prefilled value; on create, always send. The BE re-prices
     *  every existing photo under the event when this changes. */
    pricePerPhoto: number;
    /** Organizer name + race-day notes for the "About this race" strip.
     *  Trimmed; the caller forwards each only when changed on edit. */
    organizerName: string;
    description: string;
    cover: File | null;
    /** True when the user removed an existing cover and didn't pick a new
     *  file — signals to the caller to send `removeCover` to the backend.
     *  Always false in create mode (no existing cover to remove). */
    removeCover: boolean;
  }) => void | Promise<void>;
}

export function AdminEventFormModal({
  open,
  mode,
  event,
  onClose,
  onSubmit,
}: AdminEventFormModalProps) {
  const [name, setName] = useState("");
  const [location, setLocation] = useState("");
  const [date, setDate] = useState("");
  // Price is captured as a string so the user can clear / retype freely;
  // we coerce to a number at submit time. Stored separately from the
  // ListEvent.pricePerPhoto so the input doesn't churn on every catalog
  // refetch while editing.
  const [price, setPrice] = useState("");
  // Organizer name + race-day notes for the "About this race" strip. Free
  // text; trimmed at submit. Prefilled from the event in edit mode.
  const [organizerName, setOrganizerName] = useState("");
  const [description, setDescription] = useState("");
  // `coverFile` is a newly picked file. `existingBannerUrl` is the URL
  // already on the event in edit mode. Preview prefers coverFile.
  const [coverFile, setCoverFile] = useState<File | null>(null);
  const [existingBannerUrl, setExistingBannerUrl] = useState<string | undefined>(
    undefined,
  );
  // Snapshot of whether the event had a cover when the modal opened — lets
  // submit distinguish "user removed an existing cover" from "create mode
  // with no cover picked." Without this we couldn't tell which case wants
  // removeCover=true on the PATCH.
  const [initialHadCover, setInitialHadCover] = useState(false);
  const [coverError, setCoverError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Reset / prefill whenever the modal opens or the target event flips.
  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && event) {
      setName(event.name);
      setLocation(event.location);
      setDate(event.date);
      setPrice(formatPriceForInput(event.pricePerPhoto));
      setOrganizerName(event.organizerName ?? "");
      setDescription(event.description ?? "");
      setExistingBannerUrl(event.bannerUrl);
      setInitialHadCover(Boolean(event.bannerUrl));
    } else {
      setName("");
      setLocation("");
      setDate("");
      setPrice("");
      setOrganizerName("");
      setDescription("");
      setExistingBannerUrl(undefined);
      setInitialHadCover(false);
    }
    setCoverFile(null);
    setCoverError(null);
    setSubmitting(false);
  }, [open, mode, event]);

  // Object URL preview for newly picked files. Revoke on swap/close so we
  // don't leak the blob references the browser holds per createObjectURL.
  const coverPreviewUrl = useMemo(() => {
    if (!coverFile) return undefined;
    return URL.createObjectURL(coverFile);
  }, [coverFile]);
  useEffect(() => {
    return () => {
      if (coverPreviewUrl) URL.revokeObjectURL(coverPreviewUrl);
    };
  }, [coverPreviewUrl]);

  const displayedBannerUrl = coverPreviewUrl ?? existingBannerUrl;

  const trimmedName = name.trim();
  const trimmedLocation = location.trim();
  const dateValid = /^\d{4}-\d{2}-\d{2}$/.test(date);
  // Accept "", "0", "80", "80.5", "80.50" — anything that parses as a
  // non-negative finite number. Submit coerces to Number; blank means 0.
  const parsedPrice = parsePrice(price);
  const priceValid = parsedPrice !== null;
  const priceHint = !priceValid
    ? "Enter a non-negative number (e.g. 80 or 80.50)."
    : undefined;
  const canSubmit =
    trimmedName.length > 0 &&
    trimmedLocation.length > 0 &&
    dateValid &&
    priceValid &&
    !submitting;

  async function handleSubmit() {
    if (!canSubmit || parsedPrice === null) return;
    // User cleared an existing cover (Remove button) and didn't replace it.
    // The new file branch takes precedence in the parent + backend if the
    // user actually picked a replacement.
    const removeCover =
      initialHadCover && coverFile === null && existingBannerUrl === undefined;
    setSubmitting(true);
    try {
      await onSubmit({
        name: trimmedName,
        location: trimmedLocation,
        date,
        pricePerPhoto: parsedPrice,
        organizerName: organizerName.trim(),
        description: description.trim(),
        cover: coverFile,
        removeCover,
      });
    } finally {
      setSubmitting(false);
    }
  }

  const title = mode === "create" ? "Create event" : "Edit event";
  const submitLabel = submitting
    ? mode === "create"
      ? "Creating…"
      : "Saving…"
    : mode === "create"
      ? "Create event"
      : "Save changes";
  const intro =
    mode === "create"
      ? "Title, location, and date populate the event card across the app. The cover banner is optional — events without a cover fall back to the dark-text banner."
      : `Update what runners see across /events, the photographer upload picker, and the admin board.`;

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title={title}
      intro={intro}
      submitLabel={submitLabel}
      submitDisabled={!canSubmit}
    >
      <AdminTextInput
        id="event-name"
        label="Title"
        value={name}
        onChange={setName}
        maxLength={80}
        showCounter
        placeholder="Cebu Marathon 2026"
        autoFocus
      />
      <AdminTextInput
        id="event-location"
        label="Location"
        value={location}
        onChange={setLocation}
        maxLength={120}
        showCounter
        placeholder="SRP Boulevard, Cebu City"
      />
      <div className="space-y-2">
        <AdminFieldLabel htmlFor="event-date">Date</AdminFieldLabel>
        <input
          id="event-date"
          type="date"
          value={date}
          onChange={(e) => setDate(e.target.value)}
          className={DATE_INPUT_CLS}
        />
      </div>

      <AdminTextInput
        id="event-price-per-photo"
        label="Price per photo · PHP"
        value={price}
        onChange={setPrice}
        type="number"
        min={0}
        step="0.01"
        inputMode="decimal"
        prefix="₱"
        placeholder="80"
        inputClassName="tnum"
        hint={priceHint}
      />

      <AdminTextInput
        id="event-organizer"
        label="Organizer"
        value={organizerName}
        onChange={setOrganizerName}
        maxLength={120}
        showCounter
        placeholder="Cebu Runners Club"
      />

      <AdminTextarea
        id="event-description"
        label="Race day notes"
        value={description}
        onChange={setDescription}
        maxLength={600}
        rows={4}
        placeholder="Chip-timed 21K along the coastal road. Water stations every 3 km, finisher medals for all runners."
      />

      <CoverField
        bannerUrl={displayedBannerUrl}
        error={coverError}
        onPick={(file) => {
          const validationError = validateImageFile(file);
          if (validationError) {
            setCoverError(validationError);
            return;
          }
          setCoverError(null);
          setCoverFile(file);
        }}
        onRemove={() => {
          setCoverFile(null);
          setExistingBannerUrl(undefined);
          setCoverError(null);
        }}
      />
    </AdminFormModal>
  );
}

// Formats the BE's optional pricePerPhoto for the number input. Undefined
// renders as empty so the placeholder shows; 0 renders as "0" so the admin
// sees the explicit zero rather than mistaking it for "no price yet."
function formatPriceForInput(value: number | undefined): string {
  if (value === undefined || value === null) return "";
  if (!Number.isFinite(value)) return "";
  return String(value);
}

// Returns the parsed peso amount or null when the input isn't a valid
// non-negative finite number. Blank counts as 0 so the admin can leave a
// "free event" intentionally without typing a zero.
function parsePrice(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return 0;
  const num = Number(trimmed);
  if (!Number.isFinite(num) || num < 0) return null;
  return num;
}

interface CoverFieldProps {
  bannerUrl: string | undefined;
  error: string | null;
  onPick: (file: File) => void;
  onRemove: () => void;
}

function CoverField({
  bannerUrl,
  error,
  onPick,
  onRemove,
}: CoverFieldProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    onPick(file);
  }

  return (
    <div className="space-y-3">
      <AdminFieldLabel>Cover banner — optional · 4:3 · 8 MB</AdminFieldLabel>
      <div className="rounded-2xl border border-line bg-surface overflow-hidden">
        <div className="relative aspect-[4/3] bg-ink">
          {bannerUrl ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={bannerUrl}
              alt="Cover preview"
              className="absolute inset-0 w-full h-full object-cover"
            />
          ) : (
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-bone/45">
                No cover yet
              </span>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 border-t border-line">
          <label
            className={cn(
              "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink hover:text-fresh transition-colors cursor-pointer",
            )}
          >
            {bannerUrl ? "Replace cover" : "Upload cover"}
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handleChange}
              className="sr-only"
            />
          </label>
          {bannerUrl && (
            <button
              type="button"
              onClick={onRemove}
              className="font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate hover:text-error transition-colors"
            >
              Remove
            </button>
          )}
        </div>
      </div>
      {error && <AdminFieldHint>{error}</AdminFieldHint>}
    </div>
  );
}
