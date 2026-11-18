import Link from "next/link";
import { ROUTES } from "@/lib/constants";

const BASE =
  "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] transition-colors";
const LINK = `${BASE} text-slate-soft hover:text-ink`;
const LINK_ON_DARK = `${BASE} text-paper/70 hover:text-paper`;

// "Privacy · Terms" in the footers' mono kicker style.
export function LegalLinks({ onDark = false }: { onDark?: boolean }) {
  const link = onDark ? LINK_ON_DARK : LINK;
  return (
    <nav aria-label="Legal" className="flex items-center gap-3">
      <Link href={ROUTES.PRIVACY} className={link}>
        Privacy
      </Link>
      <span aria-hidden="true" className={onDark ? "text-paper/70" : "text-slate-soft"}>
        ·
      </span>
      <Link href={ROUTES.TERMS} className={link}>
        Terms
      </Link>
    </nav>
  );
}
