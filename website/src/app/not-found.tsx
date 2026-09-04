import Link from "next/link";
import { BrandLogo } from "@/components/layout/brand-logo";
import { cn } from "@/lib/utils";
import { BTN_PRIMARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center px-4">
      <BrandLogo className="mb-8 h-11 w-44" />
      <h1 className="font-hero text-6xl text-line-strong">404</h1>
      <h2 className="mt-4 font-display text-xl font-bold text-ink">
        Page Not Found
      </h2>
      <p className="mt-2 text-slate">
        The page you&apos;re looking for doesn&apos;t exist.
      </p>
      <Link href={ROUTES.HOME} className={cn(BTN_PRIMARY, BTN_SIZE.md, "mt-6")}>
        Back to Home
      </Link>
    </div>
  );
}
