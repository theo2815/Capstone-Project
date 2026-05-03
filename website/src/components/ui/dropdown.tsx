"use client";

import {
  useState,
  useRef,
  useEffect,
  type ReactNode,
  type ButtonHTMLAttributes,
} from "react";
import { cn } from "@/lib/utils";

interface DropdownProps {
  trigger: ReactNode;
  children: ReactNode;
  align?: "left" | "right";
  className?: string;
  panelClassName?: string;
  ariaLabel?: string;
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
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setIsOpen(false);
    };
    document.addEventListener("mousedown", handleClickOutside);
    document.addEventListener("keydown", handleKey);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
      document.removeEventListener("keydown", handleKey);
    };
  }, [isOpen]);

  return (
    <div ref={ref} className={cn("relative inline-block", className)}>
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
      {isOpen && (
        <div
          role="menu"
          aria-label={ariaLabel}
          onClick={() => setIsOpen(false)}
          style={{ animation: "fade-in 0.16s ease-out both" }}
          className={cn(
            "absolute z-30 mt-2 min-w-[200px] rounded-xl border border-line bg-bone py-1 shadow-[0_12px_32px_-12px_rgba(17,17,17,0.22)] origin-top",
            align === "right" ? "right-0" : "left-0",
            panelClassName,
          )}
        >
          {children}
        </div>
      )}
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
