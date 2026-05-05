"use client";

import { useUserMediaStore } from "@/store/user-media-store";
import { cn } from "@/lib/utils";

type AvatarSize = "sm" | "md" | "lg";
type AvatarTone = "ink" | "fresh";

interface AvatarDiscProps {
  name: string;
  size?: AvatarSize;
  tone?: AvatarTone;
  className?: string;
}

const SIZE_CLASS: Record<AvatarSize, string> = {
  sm: "size-9 text-[11px] font-mono uppercase tracking-[0.05em] font-semibold",
  md: "size-14 md:size-16 text-lg md:text-xl",
  lg: "size-20 md:size-24 text-2xl md:text-3xl",
};

const TONE_CLASS: Record<AvatarTone, string> = {
  ink: "bg-ink text-bone",
  fresh: "bg-fresh text-bone",
};

export function AvatarDisc({
  name,
  size = "md",
  tone = "ink",
  className,
}: AvatarDiscProps) {
  const avatar = useUserMediaStore((s) => s.avatar);
  const sizeClass = SIZE_CLASS[size];

  if (avatar) {
    return (
      <div
        aria-hidden="true"
        className={cn(
          "rounded-full overflow-hidden bg-bone-deep shrink-0",
          sizeClass,
          className,
        )}
      >
        {/* eslint-disable-next-line @next/next/no-img-element -- data-URL avatar mock; backend will return signed S3 URLs. */}
        <img
          src={avatar.dataUrl}
          alt=""
          className="size-full object-cover"
          draggable={false}
        />
      </div>
    );
  }

  const initials = getInitials(name);
  const baseFont =
    size === "sm"
      ? ""
      : "font-display font-medium tracking-tight";

  return (
    <div
      aria-hidden="true"
      className={cn(
        "rounded-full flex items-center justify-center shrink-0",
        baseFont,
        TONE_CLASS[tone],
        sizeClass,
        className,
      )}
    >
      {initials}
    </div>
  );
}

function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}
