import { type ComponentPropsWithoutRef, type ElementType, type ReactNode } from "react";
import { cn } from "@/lib/utils";

type KickerTone = "default" | "soft" | "active";
type KickerSize = "sm" | "md";

type KickerOwnProps = {
  tone?: KickerTone;
  size?: KickerSize;
  tnum?: boolean;
  className?: string;
  children: ReactNode;
};

type KickerProps<C extends ElementType> = KickerOwnProps & {
  as?: C;
} & Omit<ComponentPropsWithoutRef<C>, keyof KickerOwnProps | "as">;

// Quiet Studio mono-uppercase eyebrow. Bakes in the comfort-safe floor with
// a graduated responsive bump:
//   sm: 13px (≤390) / 14px (≥400 big phones) / 12px (≥md desktop), tracking 0.18em
//   md: 14px (≤390) / 15px (≥400 big phones) / 13px (≥md desktop), tracking 0.22em
// The 400 px breakpoint catches iPhone XR/11/12/13/14/15 (390-430 px) and most
// Android phones — bigger viewport → kickers feel relatively smaller in context,
// so they need +1 px to match the comfort threshold SE-class phones hit at 13.
// SE-class phones (≤390 px) keep the original 13 px.
// See QuickPitik Vault/website/notes/ui-pitfalls.md (2026-05-07 floor-bump entries).
// tone="default" → text-slate (~7.4:1 on bone, AA), the everyday choice.
// tone="soft"    → text-slate-soft (now #64748B → ~5.6:1 on bone, AA).
// tone="active"  → text-fresh, reserved for the one-fresh-per-viewport accent.
//
// Replaces hand-rolled chains like:
//   <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
// with:
//   <Kicker>…</Kicker>
//
// Polymorphic: pass `as={Link}` / `as="button"` / `as="p"` etc. The remaining
// props are forwarded to the underlying element (so href, onClick, type, etc.
// are typed correctly through ComponentPropsWithoutRef<C>).
const TONE_CLASS: Record<KickerTone, string> = {
  default: "text-slate",
  soft: "text-slate-soft",
  active: "text-fresh",
};

const SIZE_CLASS: Record<KickerSize, string> = {
  sm: "text-[13px] min-[400px]:text-[14px] md:text-[12px] tracking-[0.18em]",
  md: "text-[14px] min-[400px]:text-[15px] md:text-[13px] tracking-[0.22em]",
};

export function Kicker<C extends ElementType = "span">({
  as,
  tone = "default",
  size = "sm",
  tnum = false,
  className,
  children,
  ...rest
}: KickerProps<C>) {
  const Component = (as ?? "span") as ElementType;
  return (
    <Component
      className={cn(
        "font-mono uppercase",
        SIZE_CLASS[size],
        TONE_CLASS[tone],
        tnum && "tnum",
        className,
      )}
      {...rest}
    >
      {children}
    </Component>
  );
}
