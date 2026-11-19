"use client";

import { useState } from "react";
import { Modal } from "@/components/ui/modal";
import { cn } from "@/lib/utils";

// A case photo the admin can open larger before deciding on it (AX-36 RM4).
export function RemovalPhotoThumb({ src, alt, className = "size-16" }: { src: string | null | undefined; alt: string; className?: string }) {
  const [open, setOpen] = useState(false);
  if (!src) return <span className={cn("block shrink-0 rounded-lg bg-bone-deep", className)} aria-hidden="true" />;
  return (
    <>
      <button type="button" onClick={() => setOpen(true)} aria-label={`Enlarge ${alt}`}
        className={cn("block shrink-0 overflow-hidden rounded-lg bg-bone-deep", className)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} loading="lazy" className="h-full w-full object-cover" />
      </button>
      <Modal isOpen={open} onClose={() => setOpen(false)} title={alt} className="max-w-3xl">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} className="max-h-[70vh] w-full rounded-xl object-contain" />
      </Modal>
    </>
  );
}
