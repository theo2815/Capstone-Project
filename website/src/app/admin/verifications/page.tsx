"use client";

import { usePendingVerificationsCount } from "@/hooks/use-admin-users";
import { useAdminKpis } from "@/hooks/use-admin-data";

import {
  VerificationsQueue,
} from "@/components/admin/verifications-queue";
import { SellerDetailsQueue } from "@/components/admin/seller-details-queue";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { AdminSectionIntro } from "@/components/admin/admin-section-intro";
import { ROUTES } from "@/lib/constants";

export default function AdminVerificationsPage() {
  const pendingCount = usePendingVerificationsCount();
  // An older backend sends no seller-details count: that number is left out, never shown as 0.
  const sellerDetailsCount = useAdminKpis()?.pendingSellerDetails ?? null;

  return (
    <>
      <Header pendingCount={pendingCount} sellerDetailsCount={sellerDetailsCount} />
      {/* C-01: the ID checks sit above the history; slab numbers follow W41.3. */}
      <VerificationsQueue afterIncomplete={<SellerDetailsQueue number="03" keyScope />} />
    </>
  );
}

const COUNT_LINK = "underline underline-offset-4 hover:text-ink";

function Header({ pendingCount, sellerDetailsCount }: { pendingCount: number | null; sellerDetailsCount: number | null }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Verifications{pendingCount !== null && (
          <> · <Link href="#open-queue" className={COUNT_LINK}><span className="tnum">{pendingCount}</span> {pendingCount === 1 ? "profile" : "profiles"} waiting</Link></>
        )}
        {sellerDetailsCount !== null && (
          <> · <Link href="#seller-details" className={COUNT_LINK}>Seller details · <span className="tnum">{sellerDetailsCount}</span> waiting</Link></>
        )}
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Verifications.
      </h1>
      <AdminSectionIntro href={ROUTES.ADMIN_VERIFICATIONS} />
    </header>
  );
}
