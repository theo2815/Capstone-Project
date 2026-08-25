"use client";

import { useEffect, useId, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { X } from "lucide-react";
import { useScrollLock } from "@/lib/scroll-lock";
import { Kicker } from "@/components/ui/kicker";
import { cn } from "@/lib/utils";

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
  title?: string;
  children: ReactNode;
  className?: string;
  /**
   * When true, the ESC keydown listener is disabled. Use when this modal is
   * the parent of a stacked nested modal so a single ESC press only closes
   * the topmost. See QuickPitik Vault/website/notes/ui-pitfalls.md
   * (2026-05-06 stacked-modal entry).
   */
  escDisabled?: boolean;
  /**
   * Hides the default ✕ close button. Pass when the modal body renders its
   * own close affordance, or when dismissal must be gated (mid-payment).
   */
  hideClose?: boolean;
}

// Portals to document.body so the modal escapes any transform-context
// ancestor (stagger-children, fade-up animations, etc.). Quiet Studio
// surface — bone card on bg-ink/40 backdrop, mono kicker title.
export function Modal({
  isOpen,
  onClose,
  title,
  children,
  className,
  escDisabled = false,
  hideClose = false,
}: ModalProps) {
  const [mounted, setMounted] = useState(false);
  const titleId = useId();
  useScrollLock(isOpen);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!isOpen || escDisabled) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", handleEsc);
    return () => {
      document.removeEventListener("keydown", handleEsc);
    };
  }, [isOpen, escDisabled, onClose]);

  if (!isOpen || !mounted) return null;

  const content = (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-6 md:p-8">
      <div
        className="fixed inset-0 bg-ink/40 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden
      />
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={title ? titleId : undefined}
        className={cn(
          "relative z-10 w-full max-w-lg rounded-2xl border border-line bg-bone p-6 md:p-8 shadow-2xl",
          "max-h-[calc(100vh-3rem)] overflow-y-auto",
          "animate-[fade-up_0.25s_ease-out_both]",
          className,
        )}
      >
        {!hideClose && (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="absolute right-2 top-2 z-10 flex size-11 items-center justify-center rounded-full text-slate transition-colors hover:bg-bone-deep hover:text-ink"
          >
            <X className="size-4" aria-hidden="true" />
          </button>
        )}
        {title && (
          <Kicker as="h2" id={titleId} size="md" className="mb-6 pr-10">
            {title}
          </Kicker>
        )}
        {children}
      </div>
    </div>
  );

  return createPortal(content, document.body);
}
