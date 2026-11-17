import Link from "next/link";
import { BrandLogo } from "@/components/layout/brand-logo";
import { LegalLinks } from "@/components/legal/legal-links";
import { BTN_ON_DARK, BTN_SIZE } from "@/components/ui/button-styles";
import { Kicker } from "@/components/ui/kicker";
import { ROUTES, SUPPORT_EMAIL } from "@/lib/constants";
import { cn } from "@/lib/utils";

// Chequered finish-line strip along the top edge of the dark footers.
export function FinishTape() {
  return (
    <div
      aria-hidden="true"
      className="h-2.5 w-full opacity-90"
      style={{
        background:
          "repeating-linear-gradient(90deg, var(--paper) 0 18px, transparent 18px 36px)",
      }}
    />
  );
}

export function Footer({
  showCta = true,
  className,
}: {
  showCta?: boolean;
  className?: string;
}) {
  return (
    <footer className={cn("relative overflow-hidden bg-pine text-paper", className)}>
      <FinishTape />
      <div className="mx-auto max-w-7xl px-6 md:px-10">
        {showCta && (
          <div className="flex flex-col gap-6 border-b border-paper/15 py-12 md:flex-row md:items-center md:justify-between">
            <div>
              <h2 className="font-hero text-[40px] md:text-[54px]">Find your finish.</h2>
              <p className="mt-2 max-w-md text-[15px] text-paper/75">
                Race photos from Cebu events, searchable by selfie or bib minutes
                after you cross the line.
              </p>
            </div>
            <Link
              href={ROUTES.EVENTS}
              className={cn(BTN_ON_DARK, BTN_SIZE.md, "w-full md:w-auto")}
            >
              Find my photos
            </Link>
          </div>
        )}

        <div className="grid grid-cols-2 gap-x-6 gap-y-8 py-10 md:grid-cols-[1.4fr_1fr_1fr_1fr]">
          <div className="col-span-2 md:col-span-1">
            <Link
              href={ROUTES.RUNNERS}
              aria-label="QuickPitik home"
              className="inline-block rounded-lg bg-paper px-3.5 py-2.5"
            >
              <BrandLogo className="h-8 w-32" />
            </Link>
            <p className="mt-4 max-w-xs text-[15px] text-paper/75">
              Photographers get found and paid. Runners get their moment.
            </p>
          </div>

          <FooterColumn title="Runners">
            <FooterLink href={ROUTES.RUNNERS}>For runners</FooterLink>
            <FooterLink href={ROUTES.EVENTS}>Browse events</FooterLink>
            <FooterLink href={ROUTES.VERIFY}>Verify a photo</FooterLink>
          </FooterColumn>

          <FooterColumn title="Photographers">
            <FooterLink href={ROUTES.PHOTOGRAPHERS}>For photographers</FooterLink>
            <FooterLink href={`${ROUTES.REGISTER}?role=PHOTOGRAPHER`}>
              Join as a photographer
            </FooterLink>
          </FooterColumn>

          <FooterColumn title="Help" className="col-span-2 md:col-span-1">
            <FooterLink href={`mailto:${SUPPORT_EMAIL}`} className="break-all">
              {SUPPORT_EMAIL}
            </FooterLink>
          </FooterColumn>
        </div>

        <div className="flex flex-col gap-3 border-t border-paper/15 py-5 md:flex-row md:items-center md:justify-between">
          <Kicker tnum className="text-paper/60">
            {`© ${new Date().getFullYear()} QuickPitik`}
          </Kicker>
          <LegalLinks onDark />
          <Kicker className="text-paper/60">Cebu, Philippines</Kicker>
        </div>
      </div>

      <p
        aria-hidden="true"
        className="font-hero -mb-3 -ml-2 select-none whitespace-nowrap text-[84px] leading-[0.78] text-transparent md:-mb-6 md:text-[190px]"
        style={{
          WebkitTextStroke:
            "1px color-mix(in srgb, var(--paper) 16%, transparent)",
        }}
      >
        QuickPitik
      </p>
    </footer>
  );
}

function FooterColumn({
  title,
  className,
  children,
}: {
  title: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <div className={className}>
      <Kicker as="h3" className="text-fresh-tint/75">
        {title}
      </Kicker>
      <ul className="mt-2">{children}</ul>
    </div>
  );
}

function FooterLink({
  href,
  className,
  children,
}: {
  href: string;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <li>
      <Link
        href={href}
        className={cn(
          "inline-flex min-h-11 items-center text-[15px] text-paper/90 underline decoration-paper/25 underline-offset-4 hover:text-paper hover:decoration-paper md:min-h-9",
          className,
        )}
      >
        {children}
      </Link>
    </li>
  );
}
