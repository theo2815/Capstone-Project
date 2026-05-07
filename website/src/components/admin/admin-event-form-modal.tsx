"use client";

import { type ChangeEvent, useEffect, useRef, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFieldHint,
  AdminFieldLabel,
  AdminTextInput,
} from "@/components/admin/admin-form-fields";
import type { ListEvent } from "@/app/events/events-browser";
import {
  ACCEPTED_IMAGE_MIME,
  fitToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";
import { cn } from "@/lib/utils";

// Create / edit form modal for admin events. Fields: Title, Location, Date,
// and an optional cover banner. Cover is a data URL produced via
// `fitToDataUrl(file, 1920, 0.82)` — matches the photographer cover
// pipeline so size + quality stay consistent. Empty cover falls back to the
// dark-text banner already used by the event tile.
//
// Mode prop selects copy + button label. In `edit` mode the form prefills
// from the passed `event`; the date input enforces ISO YYYY-MM-DD.

const DATE_INPUT_CLS =
  "w-full rounded-2xl border border-line bg-surface px-4 py-3 font-sans text-sm text-ink focus:outline-none focus:ring-2 focus:ring-fresh focus:border-fresh tnum";

const COVER_MAX_PX = 1920;

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
    bannerUrl?: string;
  }) => void;
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
  const [bannerUrl, setBannerUrl] = useState<string | undefined>(undefined);
  const [coverBusy, setCoverBusy] = useState(false);
  const [coverError, setCoverError] = useState<string | null>(null);

  // Reset / prefill whenever the modal opens or the target event flips.
  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && event) {
      setName(event.name);
      setLocation(event.location);
      setDate(event.date);
      setBannerUrl(event.bannerUrl);
    } else {
      setName("");
      setLocation("");
      setDate("");
      setBannerUrl(undefined);
    }
    setCoverError(null);
    setCoverBusy(false);
  }, [open, mode, event]);

  const trimmedName = name.trim();
  const trimmedLocation = location.trim();
  const dateValid = /^\d{4}-\d{2}-\d{2}$/.test(date);
  const canSubmit =
    trimmedName.length > 0 &&
    trimmedLocation.length > 0 &&
    dateValid &&
    !coverBusy;

  function handleSubmit() {
    if (!canSubmit) return;
    onSubmit({
      name: trimmedName,
      location: trimmedLocation,
      date,
      bannerUrl,
    });
  }

  const title = mode === "create" ? "Create event" : "Edit event";
  const submitLabel = mode === "create" ? "Create event" : "Save changes";
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

      <CoverField
        bannerUrl={bannerUrl}
        busy={coverBusy}
        error={coverError}
        onPick={async (file) => {
          const validationError = validateImageFile(file);
          if (validationError) {
            setCoverError(validationError);
            return;
          }
          setCoverBusy(true);
          setCoverError(null);
          try {
            const dataUrl = await fitToDataUrl(file, COVER_MAX_PX, 0.82);
            setBannerUrl(dataUrl);
          } catch {
            setCoverError("Could not process this image. Try another.");
          } finally {
            setCoverBusy(false);
          }
        }}
        onRemove={() => {
          setBannerUrl(undefined);
          setCoverError(null);
        }}
      />
    </AdminFormModal>
  );
}

interface CoverFieldProps {
  bannerUrl: string | undefined;
  busy: boolean;
  error: string | null;
  onPick: (file: File) => void;
  onRemove: () => void;
}

function CoverField({
  bannerUrl,
  busy,
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
              <span className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-bone/45">
                No cover yet
              </span>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 border-t border-line">
          <label
            className={cn(
              "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors",
              busy ? "opacity-60 cursor-wait" : "cursor-pointer",
            )}
          >
            {busy ? "Processing…" : bannerUrl ? "Replace cover" : "Upload cover"}
            <input
              ref={inputRef}
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handleChange}
              disabled={busy}
              className="sr-only"
            />
          </label>
          {bannerUrl && !busy && (
            <button
              type="button"
              onClick={onRemove}
              className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-slate hover:text-error transition-colors"
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
