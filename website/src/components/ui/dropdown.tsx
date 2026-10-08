"use client";

import {
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  useCallback,
  type ReactNode,
  type ButtonHTMLAttributes,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";
import { KICKER_SIZE_CLASS } from "@/components/ui/kicker";

// The panel portals to document.body so it escapes ancestor stacking
// contexts. `stagger-children` + `fade-up` (animation-fill-mode: both) leaves
// each animated element with a persisted `transform: translateY(0)`, which
// makes every animated sibling its own stacking context — meaning a `z-30`
// panel inside Slab N still paints below Slab N+1's content. Same trap that
// motivated the PhotoPreviewCard portal fix on /orders. See
// `notes/ui-pitfalls.md` for the full root cause.

interface DropdownProps {
  trigger: ReactNode;
  children: ReactNode;
  align?: "left" | "right";
  className?: string;
  panelClassName?: string;
  ariaLabel?: string;
  id?: string;
  ariaDescribedBy?: string;
  disabled?: boolean;
  onOpenChange?: (open: boolean) => void;
}

interface PanelPosition {
  top: number;
  left: number;
  triggerWidth: number;
  maxHeight: number;
}

export function Dropdown({
  trigger,
  children,
  align = "left",
  className,
  panelClassName,
  ariaLabel,
  id,
  ariaDescribedBy,
  disabled = false,
  onOpenChange,
}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [position, setPosition] = useState<PanelPosition>({
    top: 0,
    left: 0,
    triggerWidth: 0,
    maxHeight: 320,
  });
  const triggerWrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const expanded = isOpen && !disabled;

  useEffect(() => { onOpenChange?.(expanded); }, [expanded, onOpenChange]);
  useEffect(() => { if (disabled) setIsOpen(false); }, [disabled]);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Recompute panel position from the trigger's viewport rect. Called on
  // open + scroll + resize so the panel stays anchored to the trigger.
  const updatePosition = useCallback(() => {
    const wrap = triggerWrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    const panel = panelRef.current?.getBoundingClientRect();
    const width = Math.min(Math.max(panel?.width ?? 200, rect.width), window.innerWidth - 16);
    const height = panel?.height || 320;
    const below = Math.max(0, window.innerHeight - rect.bottom - 16);
    const above = Math.max(0, rect.top - 16);
    const openAbove = below < height && above > below;
    setPosition({
      top: openAbove ? rect.top - 8 - Math.min(height, above) : rect.bottom + 8,
      left: Math.max(8, Math.min(align === "right" ? rect.right - width : rect.left, window.innerWidth - width - 8)),
      triggerWidth: Math.min(rect.width, window.innerWidth - 16),
      maxHeight: Math.min(320, openAbove ? above : below),
    });
  }, [align]);

  useLayoutEffect(() => {
    if (!isOpen) return;
    updatePosition();
  }, [isOpen, updatePosition]);

  useEffect(() => {
    if (!isOpen) return;

    const handleClickOutside = (e: MouseEvent) => {
      const target = e.target as Node;
      if (
        triggerWrapRef.current?.contains(target) ||
        panelRef.current?.contains(target)
      ) {
        return;
      }
      setIsOpen(false);
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setIsOpen(false);
        triggerWrapRef.current?.querySelector("button")?.focus();
      }
    };
    const handleScroll = () => updatePosition();
    const handleResize = () => updatePosition();

    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKey);
    // Capture phase catches scroll on any ancestor (including the page body).
    window.addEventListener("scroll", handleScroll, true);
    window.addEventListener("resize", handleResize);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKey);
      window.removeEventListener("scroll", handleScroll, true);
      window.removeEventListener("resize", handleResize);
    };
  }, [isOpen, updatePosition]);

  const panel =
    expanded && mounted
      ? createPortal(
          <div
            ref={panelRef}
            role="menu"
            aria-label={ariaLabel}
            onClick={() => setIsOpen(false)}
            style={{
              animation: "fade-in 0.16s ease-out both",
              position: "fixed",
              top: position.top,
              left: position.left,
              minWidth: position.triggerWidth,
              maxWidth: "calc(100vw - 16px)",
              maxHeight: position.maxHeight,
              overflowY: "auto",
            }}
            className={cn(
              "z-50 min-w-[200px] rounded-xl border border-line bg-bone py-1 shadow-[0_12px_32px_-12px_rgba(17,17,17,0.22)] origin-top",
              panelClassName,
            )}
          >
            {children}
          </div>,
          document.body,
        )
      : null;

  return (
    <div
      ref={triggerWrapRef}
      className={cn("relative inline-block", className)}
    >
      <button
        type="button"
        id={id}
        disabled={disabled}
        aria-describedby={ariaDescribedBy}
        onClick={() => setIsOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={expanded}
        aria-label={ariaLabel}
        className="rounded-sm disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
      >
        {trigger}
      </button>
      {panel}
    </div>
  );
}

interface DropdownTriggerProps {
  children: ReactNode;
  className?: string;
}

/** Convenience trigger styled to match Quiet Studio mono-uppercase kickers. */
export function DropdownTrigger({ children, className }: DropdownTriggerProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 font-mono uppercase text-ink hover:text-fresh transition-colors",
        KICKER_SIZE_CLASS.sm,
        className,
      )}
    >
      <span className="truncate">{children}</span>
      <svg
        className="size-3 shrink-0 text-slate"
        viewBox="0 0 16 16"
        fill="none"
        aria-hidden="true"
      >
        <path
          d="M4 6 L8 10 L12 6"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    </span>
  );
}

interface DropdownItemProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  active?: boolean;
  variant?: "default" | "mono";
}

export function DropdownItem({
  className,
  children,
  active = false,
  variant = "default",
  ...props
}: DropdownItemProps) {
  return (
    <button
      type="button"
      role="menuitem"
      aria-current={active ? "true" : undefined}
      className={cn(
        "flex w-full items-center justify-between gap-3 px-4 py-2.5 transition-colors hover:bg-bone-deep focus:bg-bone-deep focus:outline-none",
        variant === "mono"
          ? cn("font-mono uppercase", KICKER_SIZE_CLASS.sm)
          : "font-sans text-sm",
        active ? "text-fresh" : "text-ink",
        className,
      )}
      {...props}
    >
      <span className="truncate text-left">{children}</span>
      {active && (
        <span
          className="size-1.5 shrink-0 rounded-full bg-fresh"
          aria-hidden="true"
        />
      )}
    </button>
  );
}
