import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

// FT-4: inside the admin layout, so a bad admin link keeps the rail and offers the Inbox.
export default function AdminNotFound() {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">Admin</Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Not found.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        This admin page does not exist. The link may be old, or the item may have been removed.
      </p>
      <Link href={ROUTES.ADMIN_INBOX} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-6")}>
        Go to the Inbox
      </Link>
    </header>
  );
}
