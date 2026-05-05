"use client";

import { useState, type ChangeEvent } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useUserMediaStore } from "@/store/user-media-store";
import { useToast } from "@/hooks/use-toast";
import {
  ACCEPTED_IMAGE_MIME,
  squareCropToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";
import { AvatarDisc } from "@/components/account/avatar-disc";

const AVATAR_SIZE_PX = 512;

export function AvatarSlab() {
  const { user } = useAuth();
  const avatar = useUserMediaStore((s) => s.avatar);
  const setAvatar = useUserMediaStore((s) => s.setAvatar);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!user) return null;

  async function handlePick(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;

    const validationError = validateImageFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setBusy(true);
    setError(null);
    try {
      // TODO(backend): swap for `api.post("/me/avatar", formData)`. Server-side
      // will do the same center-crop + JPEG re-encode this helper does.
      const dataUrl = await squareCropToDataUrl(file, AVATAR_SIZE_PX);
      setAvatar({ dataUrl, uploadedAt: new Date().toISOString() });
      showToast({ kind: "success", message: "Profile picture updated." });
    } catch {
      setError("Could not process this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  function handleRemove() {
    // TODO(backend): swap for `api.delete("/me/avatar")`.
    setAvatar(null);
    setError(null);
    showToast({ kind: "success", message: "Profile picture removed." });
  }

  return (
    <div className="flex flex-col md:flex-row items-start gap-6 md:gap-10">
      <AvatarDisc name={user.name} size="lg" />
      <div className="flex-1 min-w-0 space-y-5">
        <div>
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink">
            {avatar ? "Looking sharp." : "Add a profile picture."}
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-md">
            Shown next to your name across QuickPitik. Square crop, 512×512.
            Different from your face-search selfies — those live in your{" "}
            <a
              href="/profile#selfies"
              className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
            >
              selfie library
            </a>
            .
          </p>
        </div>

        {error && (
          <p role="alert" className="font-sans text-sm text-error">
            {error}
          </p>
        )}

        <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
          <label
            className={
              "font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2 " +
              (busy ? "opacity-60 cursor-wait" : "cursor-pointer")
            }
          >
            {busy ? "Processing…" : avatar ? "Replace picture" : "Upload picture"}
            {!busy && <span aria-hidden="true">→</span>}
            <input
              type="file"
              accept={ACCEPTED_IMAGE_MIME.join(",")}
              onChange={handlePick}
              disabled={busy}
              className="sr-only"
            />
          </label>

          {avatar && !busy && (
            <button
              type="button"
              onClick={handleRemove}
              className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
            >
              Remove
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
