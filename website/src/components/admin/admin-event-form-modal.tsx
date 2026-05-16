"use client";

import { type ChangeEvent, useEffect, useMemo, useRef, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminFieldHint,
  AdminFieldLabel,
  AdminTextInput,
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
      setExistingBannerUrl(event.bannerUrl);
      setInitialHadCover(Boolean(event.bannerUrl));
    } else {
      setName("");
      setLocation("");
      setDate("");
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
  const canSubmit =
    trimmedName.length > 0 &&
    trimmedLocation.length > 0 &&
    dateValid &&
    !submitting;

  async function handleSubmit() {
    if (!canSubmit) return;
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
              <span className="font-mono uppercase tracking-[0.3em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-bone/45">
                No cover yet
              </span>
            </div>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-x-5 gap-y-2 px-4 py-3 border-t border-line">
          <label
            className={cn(
              "font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink hover:text-fresh transition-colors cursor-pointer",
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
