"use client";

import { useMemo } from "react";
import { Kicker } from "@/components/ui/kicker";
import { formatPayoutNumber } from "@/lib/payout-format";
import { payoutMethodLabel, type AdminPayoutCycle } from "@/lib/admin-payouts";
import { isMockQrUrl, mockQrSvgString } from "@/lib/mock-qr";
import { formatPrice } from "@/lib/utils";

// "TRANSFER TO" reference card — surfaces the photographer's payout account
// snapshot at the moment admin needs it. Two render sites:
//   1. Mark-paid modal (mode="modal") — also shows the QR code and a
//      Download QR button so admin can transfer via QR-scan.
//   2. Drawer (mode="drawer") — compact text-only variant, used while
//      admin is reading context before clicking Mark paid.
//
// Account number is always rendered in full (formatPayoutNumber) — admin
// needs the raw digits to type into their banking app or compare against
// the QR scan result.

interface PayoutAccountCardProps {
  cycle: AdminPayoutCycle;
  mode?: "modal" | "drawer";
}

export function PayoutAccountCard({
  cycle,
  mode = "drawer",
}: PayoutAccountCardProps) {
  const { payoutAccount, amount } = cycle;
  const formattedNumber = formatPayoutNumber(
    payoutAccount.method,
    payoutAccount.accountNumber,
  );

  const containerClass =
    mode === "modal"
      ? "rounded-xl border border-line bg-bone-deep px-4 py-4"
      : "rounded-2xl border border-line bg-bone-deep p-4";

  return (
    <div className={containerClass}>
      <Kicker as="p" tone="soft">
        Transfer to
      </Kicker>
      <p className="font-sans text-base text-ink mt-1">
        {payoutAccount.accountName}
      </p>
      <p className="font-mono text-sm text-ink tnum mt-1 break-all">
        {payoutMethodLabel(payoutAccount.method)} · {formattedNumber}
      </p>
      <Kicker as="p" tone="soft" tnum className="mt-2">
        Amount due · {formatPrice(amount)}
      </Kicker>
      {mode === "modal" && payoutAccount.qr && (
        <PayoutQrBlock
          qr={payoutAccount.qr}
          accountName={payoutAccount.accountName}
          accountNumber={payoutAccount.accountNumber}
          method={payoutMethodLabel(payoutAccount.method)}
          cycleId={cycle.id}
        />
      )}
    </div>
  );
}

interface PayoutQrBlockProps {
  qr: { dataUrl: string; uploadedAt: string };
  accountName: string;
  accountNumber: string;
  method: string;
  cycleId: string;
}

function PayoutQrBlock({
  qr,
  accountName,
  accountNumber,
  method,
  cycleId,
}: PayoutQrBlockProps) {
  const isMock = isMockQrUrl(qr.dataUrl);
  const payload = `${method}:${accountNumber}:${accountName}`;

  // For mock placeholders, render the SVG inline. For real uploads (Phase F)
  // the dataUrl is a data: URI — render as <img>.
  const svgString = useMemo(
    () => (isMock ? mockQrSvgString(payload, 6) : null),
    [isMock, payload],
  );

  function handleDownload() {
    const fileSafeName = accountName.replace(/[^a-z0-9]+/gi, "-").toLowerCase();
    const fileName = `qr-${cycleId}-${fileSafeName}`;
    if (svgString) {
      const blob = new Blob([svgString], { type: "image/svg+xml" });
      const url = URL.createObjectURL(blob);
      triggerDownload(url, `${fileName}.svg`);
      setTimeout(() => URL.revokeObjectURL(url), 200);
    } else {
      triggerDownload(qr.dataUrl, `${fileName}.png`);
    }
  }

  return (
    <div className="mt-4 pt-4 border-t border-line">
      <Kicker as="p" tone="soft" className="mb-3">
        Or scan QR
      </Kicker>
      <div className="flex flex-col items-center gap-3">
        <div className="size-44 rounded-xl border border-line bg-bone p-2 flex items-center justify-center">
          {svgString ? (
            <div
              className="size-full"
              role="img"
              aria-label={`Payment QR code for ${accountName}`}
              dangerouslySetInnerHTML={{ __html: svgString }}
            />
          ) : (
            // eslint-disable-next-line @next/next/no-img-element -- mock data URL
            <img
              src={qr.dataUrl}
              alt={`Payment QR code for ${accountName}`}
              className="size-full object-contain"
              draggable={false}
            />
          )}
        </div>
        <button
          type="button"
          onClick={handleDownload}
          className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-line hover:bg-ink hover:text-bone hover:border-ink transition-colors rounded-full px-5 py-2 inline-flex items-center gap-2"
        >
          Download QR
          <span aria-hidden="true">↓</span>
        </button>
      </div>
    </div>
  );
}

function triggerDownload(href: string, fileName: string): void {
  const a = document.createElement("a");
  a.href = href;
  a.download = fileName;
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}
