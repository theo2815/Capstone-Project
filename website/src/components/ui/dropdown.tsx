"use client";

import {
  useState,
  useRef,
  useEffect,
  useLayoutEffect,
  type ReactNode,
  type ButtonHTMLAttributes,
} from "react";
import { createPortal } from "react-dom";
import { cn } from "@/lib/utils";

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
}

interface PanelPosition {
  top: number;
  left: number;
  rightFromViewport: number;
  triggerWidth: number;
}

export function Dropdown({
  trigger,
  children,
  align = "left",
  className,
  panelClassName,
  ariaLabel,
}: DropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [mounted, setMounted] = useState(false);
  const [position, setPosition] = useState<PanelPosition>({
    top: 0,
    left: 0,
    rightFromViewport: 0,
    triggerWidth: 0,
  });
  const triggerWrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Recompute panel position from the trigger's viewport rect. Called on
  // open + scroll + resize so the panel stays anchored to the trigger.
  function updatePosition() {
    const wrap = triggerWrapRef.current;
    if (!wrap) return;
    const rect = wrap.getBoundingClientRect();
    setPosition({
      top: rect.bottom + 8,
      left: rect.left,
      rightFromViewport: window.innerWidth - rect.right,
      triggerWidth: rect.width,
    });
  }

  useLayoutEffect(() => {
    if (!isOpen) return;
    updatePosition();
  }, [isOpen]);

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
      if (e.key === "Escape") setIsOpen(false);
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
  }, [isOpen]);

  const panel =
    isOpen && mounted
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
              ...(align === "right"
                ? { right: position.rightFromViewport }
                : { left: position.left }),
              minWidth: position.triggerWidth,
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
        onClick={() => setIsOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        aria-label={ariaLabel}
        className="rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
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
        "inline-flex items-center gap-2 font-mono uppercase tracking-[0.25em] text-[10px] text-ink hover:text-fresh transition-colors",
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
          ? "font-mono uppercase tracking-[0.25em] text-[10px]"
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
