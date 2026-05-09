"use client";

import Link from "next/link";
import { useState, type ChangeEvent } from "react";
import { useAuth } from "@/hooks/use-auth";
import { useAuthStore } from "@/store/auth-store";
import { useUserMediaStore } from "@/store/user-media-store";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import { uploadAvatar, deleteAvatar } from "@/lib/api-avatar";
import { BACKEND_LIVE } from "@/lib/backend-flag";
import {
  ACCEPTED_IMAGE_MIME,
  squareCropToDataUrl,
  validateImageFile,
} from "@/lib/image-utils";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { FieldError } from "@/components/ui/field-error";

const AVATAR_SIZE_PX = 512;

export function AvatarSlab() {
  const { user } = useAuth();
  const setUser = useAuthStore((s) => s.setUser);
  const storeAvatar = useUserMediaStore((s) => s.avatar);
  const setAvatar = useUserMediaStore((s) => s.setAvatar);
  const { showToast } = useToast();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!user) return null;

  // In live mode, the canonical avatar lives on `user.avatarUrl`. The store
  // remains the mock-mode source-of-truth; the union here lets the "Replace /
  // Remove" CTAs surface as soon as either source has an image.
  const hasAvatar = BACKEND_LIVE ? !!user.avatarUrl : !!storeAvatar;

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
      if (BACKEND_LIVE) {
        const updated = await uploadAvatar(file);
        setUser(updated);
        // Drop any leftover mock-mode store avatar so the UI tracks user.avatarUrl.
        if (storeAvatar) setAvatar(null);
      } else {
        const dataUrl = await squareCropToDataUrl(file, AVATAR_SIZE_PX);
        setAvatar({ dataUrl, uploadedAt: new Date().toISOString() });
      }
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
    if (BACKEND_LIVE) {
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
    } else {
      setAvatar(null);
    }
    showToast({ kind: "success", message: "Profile picture removed." });
  }

  return (
    <div className="flex flex-col md:flex-row items-start gap-6 md:gap-10">
      <AvatarDisc name={user.name} size="lg" />
      <div className="flex-1 min-w-0 space-y-5">
        <div>
          <p className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink">
            {hasAvatar ? "Looking sharp." : "Add a profile picture."}
          </p>
          <p className="font-sans text-sm text-slate mt-2 max-w-md">
            Shown next to your name across QuickPitik. Square crop, 512×512.
            Different from your face-search selfies — those live in your{" "}
            <Link
              href="/profile#selfies"
              className="text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
            >
              selfie library
            </Link>
            .
          </p>
        </div>

        <FieldError message={error} id="account-avatar-error" density="tight" />

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
      </div>
    </div>
  );
}
