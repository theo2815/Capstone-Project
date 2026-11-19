"use client";

import { usePendingVerificationsCount } from "@/hooks/use-admin-users";
import { useAdminKpis } from "@/hooks/use-admin-data";

import {
  VerificationsQueue,
} from "@/components/admin/verifications-queue";
import { SellerDetailsQueue } from "@/components/admin/seller-details-queue";
import { Kicker } from "@/components/ui/kicker";

export default function AdminVerificationsPage() {
  const pendingCount = usePendingVerificationsCount();
  // An older backend sends no seller-details count: that number is left out, never shown as 0.
  const sellerDetailsCount = useAdminKpis()?.pendingSellerDetails ?? null;

  return (
    <>
      <Header pendingCount={pendingCount} sellerDetailsCount={sellerDetailsCount} />
      <VerificationsQueue />
      <SellerDetailsQueue />
    </>
  );
}

function Header({ pendingCount, sellerDetailsCount }: { pendingCount: number | null; sellerDetailsCount: number | null }) {
  return (
    <header className="pb-8 md:pb-12 border-b border-line">
      <Kicker as="p">
        Profiles{pendingCount !== null && <> · <span className="tnum">{pendingCount}</span> waiting</>}
        {sellerDetailsCount !== null && (
          <> · Seller details · <span className="tnum">{sellerDetailsCount}</span> to review</>
        )}
      </Kicker>
      <h1 className="font-display text-3xl md:text-4xl font-extrabold tracking-tight leading-[1.05] text-ink mt-3">
        Verifications.
      </h1>
      <p className="font-sans text-sm md:text-base text-ink-soft mt-3 max-w-xl">
        Profiles go live by themselves once the identity check passes. Here
        you reject a profile, or approve one that was held.
      </p>
    </header>
  );
}
