import Image from "next/image";
import { cn } from "@/lib/utils";

interface BrandLogoProps {
  className?: string;
  priority?: boolean;
}

export function BrandLogo({ className, priority = false }: BrandLogoProps) {
  return (
    <span
      className={cn(
        "relative block h-10 w-[160px] shrink-0 overflow-hidden",
        className,
      )}
    >
      <Image
        src="/brand/quickpitik-logo-transparent.png"
        alt="QuickPitik"
        fill
        priority={priority}
        sizes="(min-width: 768px) 192px, 160px"
        className="select-none object-contain object-center"
        draggable={false}
      />
    </span>
  );
}
