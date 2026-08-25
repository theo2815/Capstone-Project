"use client";

import { useState } from "react";
import { Kicker } from "@/components/ui/kicker";
import { useToast } from "@/hooks/use-toast";
import { type AdminPayoutCycle, payoutMethodLabel } from "@/lib/admin-payouts";
import { formatPayoutNumber } from "@/lib/payout-format";
import { formatPrice } from "@/lib/utils";
import type { PayoutMethod } from "@/store/photographer-settings-store";

// Focal card for the approved-state drawer. The admin has approved the cycle
// — money still has to leave their bank by hand. This card surfaces the
// three things needed to do that transfer:
//   1. How much (big tnum amount, fresh).
//   2. Where (account name + method + raw digits in mono, copy-to-clipboard).
//   3. A jump-out into the right payment web app for the method.
// Mark-paid stays in the drawer footer — this card just nudges the admin
// toward it once the transfer is done.

const METHOD_WEB: Record<
  PayoutMethod,
  { url: string; label: string }
> = {
  gcash: { url: "https://www.gcash.com", label: "Open GCash" },
  maya: { url: "https://maya.ph", label: "Open Maya" },
  gotyme: { url: "https://www.gotyme.com.ph", label: "Open GoTyme" },
};

export function ReadyToSendCard({ cycle }: { cycle: AdminPayoutCycle }) {
  const { payoutAccount, amount } = cycle;
  const hasAccount = payoutAccount.accountNumber.length > 0;

  if (!hasAccount) {
    return <MissingAccountCard />;
  }

  const formattedNumber = formatPayoutNumber(
    payoutAccount.method,
    payoutAccount.accountNumber,
  );
  const methodEntry = METHOD_WEB[payoutAccount.method];

  return (
    <section className="rounded-2xl border border-fresh/40 bg-fresh/[0.04] p-5 md:p-6">
      <Kicker as="p" tone="soft">
        Ready to send
      </Kicker>
      <p className="font-display text-4xl md:text-5xl font-semibold tracking-tight text-fresh tnum mt-3 leading-none">
        {formatPrice(amount)}
      </p>

      <div className="mt-5 pt-5 border-t border-line/60 space-y-3">
        <div>
          <Kicker as="p" tone="soft">
            To
          </Kicker>
          <p className="font-sans text-base md:text-lg text-ink mt-1">
            {payoutAccount.accountName}
          </p>
          <p className="font-mono text-sm text-ink tnum mt-1">
            {payoutMethodLabel(payoutAccount.method)}
            <span className="text-slate-soft"> · </span>
            <span className="break-all">{formattedNumber}</span>
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 pt-1">
          <CopyButton value={payoutAccount.accountNumber} />
          {methodEntry && (
            <a
              href={methodEntry.url}
              target="_blank"
              rel="noopener noreferrer"
              className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-surface bg-fresh hover:bg-fresh-deep transition-colors rounded-full px-5 py-2 inline-flex items-center gap-2"
            >
              {methodEntry.label}
              <span aria-hidden="true">↗</span>
            </a>
          )}
        </div>
      </div>

      <p className="font-sans text-sm text-slate-soft mt-5">
        Transfer the amount above, then click <span className="text-ink">Mark paid</span> below and paste the bank reference.
      </p>
    </section>
  );
}

function CopyButton({ value }: { value: string }) {
  const { showToast } = useToast();
  const [copied, setCopied] = useState(false);

  async function handleCopy() {
    if (typeof navigator === "undefined" || !navigator.clipboard) {
      showToast({ kind: "error", message: "Clipboard unavailable" });
      return;
    }
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      showToast({ kind: "success", message: "Account number copied." });
      setTimeout(() => setCopied(false), 2500);
    } catch {
      showToast({ kind: "error", message: "Copy failed. Try again." });
    }
  }

  return (
    <button
      type="button"
      onClick={handleCopy}
      className="font-mono uppercase tracking-[0.25em] text-[13px] min-[400px]:text-[14px] md:text-[12px] text-ink border border-ink hover:bg-ink hover:text-surface transition-colors rounded-full px-5 py-2"
    >
      {copied ? "Copied!" : "Copy number"}
    </button>
  );
}

function MissingAccountCard() {
  return (
    <section className="rounded-2xl border border-line bg-bone-deep p-5 md:p-6">
      <Kicker as="p" tone="soft">
        Ready to send
      </Kicker>
      <p className="font-sans text-base text-ink mt-3 max-w-md">
        This photographer hasn&apos;t configured a payout account yet. Hold the
        cycle and message them to add one before transferring.
      </p>
    </section>
  );
}
