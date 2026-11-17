import Link from "next/link";
import { ROUTES } from "@/lib/constants";

const LINK =
  "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft hover:text-ink transition-colors";

// "Privacy · Terms" in the footers' mono kicker style.
export function LegalLinks() {
  return (
    <nav aria-label="Legal" className="flex items-center gap-3">
      <Link href={ROUTES.PRIVACY} className={LINK}>
        Privacy
      </Link>
      <span aria-hidden="true" className="text-slate-soft">
        ·
      </span>
      <Link href={ROUTES.TERMS} className={LINK}>
        Terms
      </Link>
    </nav>
  );
}
