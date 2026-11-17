import { BrandLogo } from "@/components/layout/brand-logo";
import { cn } from "@/lib/utils";

// "No picture yet" stand-in: the QuickPitik wordmark, softened, on a quiet
// ground, instead of colour blocks or gradients. `dark` keeps an ink ground
// where light overlay text (status chips on event tiles) sits on top.
// Pass `label` when the placeholder stands for a real image (it then reads as
// that image to screen readers); without it the placeholder is decorative.
const LOGO_SIZE = {
  sm: "h-4 w-[64px]",
  md: "h-7 w-[112px]",
  lg: "h-10 w-[160px]",
} as const;

export function LogoPlaceholder({
  tone = "light",
  size = "md",
  label,
  className,
}: {
  tone?: "light" | "dark";
  size?: keyof typeof LOGO_SIZE;
  label?: string;
  className?: string;
}) {
  return (
    <div
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      className={cn(
        "flex items-center justify-center",
        tone === "dark" ? "bg-ink" : "bg-bone-deep",
        className,
      )}
    >
      {/* The wordmark's "Quick" is black, so on ink it renders as a soft white silhouette. */}
      <span aria-hidden="true" className={tone === "dark" ? "brightness-0 invert opacity-30" : "opacity-40"}>
        <BrandLogo className={LOGO_SIZE[size]} />
      </span>
    </div>
  );
}
