"use client";

import Link from "next/link";
import { useState, type ChangeEvent } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useUserMediaStore } from "@/store/user-media-store";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { uploadAvatar, deleteAvatar } from "@/lib/api-avatar";
import {
  ACCEPTED_IMAGE_MIME,
  squareCropToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { FieldError } from "@/components/ui/field-error";

// Preview is 256px so the lg disc (96px @ 2x DPR) stays crisp without
// carrying a full-resolution data URL through React state.
const PREVIEW_PX = 256;

export function AvatarSlab() {
  const { user } = useAuth();
  const setUser = useAuthStore((s) => s.setUser);
  const storeAvatar = useUserMediaStore((s) => s.avatar);
  const setAvatar = useUserMediaStore((s) => s.setAvatar);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Picking a file no longer uploads it. The server crops to a 512 square, and
  // a phone portrait loses its top and bottom to that crop — the user should
  // see the result before it becomes their avatar, not after.
  const [pending, setPending] = useState<{
    file: File;
    previewUrl: string;
  } | null>(null);

  if (!user) return null;

  const hasAvatar = !!user.avatarUrl;

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
      const previewUrl = await squareCropToDataUrl(file, PREVIEW_PX);
      setPending({ file, previewUrl });
    } catch {
      setPending(null);
      setError("Could not read this image. Try another.");
    } finally {
      setBusy(false);
    }
  }

  async function handleConfirm() {
    if (!pending) return;
    setBusy(true);
    setError(null);
    try {
      // The original file, not the 256px preview — the backend owns the
      // authoritative 512x512 crop and sending the downscale would ship a
      // visibly softer avatar.
      const updated = await uploadAvatar(pending.file);
      setUser(updated);
      if (storeAvatar) setAvatar(null);
      setPending(null);
      showToast({ kind: "success", message: "Profile picture updated." });
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.errors[0]?.message ?? "Could not save the picture.");
      } else {
        setError("Could not process this image. Try another.");
      }
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove() {
    setError(null);
    try {
      const updated = await deleteAvatar();
      setUser(updated);
      if (storeAvatar) setAvatar(null);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? (err.errors[0]?.message ?? "Could not remove the picture.")
          : "Could not remove the picture. Try again.",
      );
      return;
    }
    showToast({ kind: "success", message: "Profile picture removed." });
  }

  return (
    <div className="flex flex-col md:flex-row items-start gap-6 md:gap-10">
      {/* Same disc, same mask, same size the real avatar will occupy — so the
          preview is a rehearsal, not an approximation. */}
      <AvatarDisc
        name={user.name}
        size="lg"
        avatarOverride={pending ? { dataUrl: pending.previewUrl } : undefined}
      />
      <div className="flex-1 min-w-0 space-y-5">
        <div>
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink">
            {pending
              ? "This is how it'll appear."
              : hasAvatar
                ? "Looking sharp."
                : "Add a profile picture."}
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-md">
            {pending ? (
              <>
                We take the centre square. If it&apos;s cropping something you
                want to keep, pick a photo that&apos;s already square.
              </>
            ) : (
              <>
                Shown next to your name across QuickPitik. Square crop, 512×512.
                Different from your face-search selfies — those live in your{" "}
                <Link
                  href="/profile#selfies"
                  className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
                >
                  selfie library
                </Link>
                .
              </>
            )}
          </p>
        </div>

        <FieldError message={error} id="account-avatar-error" density="tight" />

        {pending ? (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <button
              type="button"
              onClick={handleConfirm}
              disabled={busy}
              className="font-sans text-base font-medium bg-fresh hover:bg-fresh-deep text-bone py-3 px-6 rounded-full transition-colors disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center gap-2"
            >
              {busy ? "Saving…" : "Use this picture"}
              {!busy && <span aria-hidden="true">→</span>}
            </button>
            <label
              className={
                "font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors " +
                (busy ? "opacity-60 cursor-wait" : "cursor-pointer")
              }
            >
              Choose another
              <input
                type="file"
                accept={ACCEPTED_IMAGE_MIME.join(",")}
                onChange={handlePick}
                disabled={busy}
                className="sr-only"
              />
            </label>
            {!busy && (
              <button
                type="button"
                onClick={() => {
                  setPending(null);
                  setError(null);
                }}
                className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-ink hover:text-ink transition-colors"
              >
                Cancel
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-wrap items-center gap-x-6 gap-y-3">
            <label
              className={
                "font-sans text-base font-medium border border-ink text-ink hover:bg-ink hover:text-bone py-3 px-6 rounded-full transition-colors inline-flex items-center gap-2 " +
                (busy ? "opacity-60 cursor-wait" : "cursor-pointer")
              }
            >
              {busy ? "Processing…" : hasAvatar ? "Replace picture" : "Upload picture"}
              {!busy && <span aria-hidden="true">→</span>}
              <input
                type="file"
                accept={ACCEPTED_IMAGE_MIME.join(",")}
                onChange={handlePick}
                disabled={busy}
                className="sr-only"
              />
            </label>

            {hasAvatar && !busy && (
              <button
                type="button"
                onClick={handleRemove}
                className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors"
              >
                Remove
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
