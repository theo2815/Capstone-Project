"use client";

import { cn } from "@/lib/utils";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";

// Sits inside the admin layout, so a page that throws keeps the rail and shell.
export default function AdminError({ retry }: { error: Error & { digest?: string }; retry: () => void }) {
  return (
    <div role="alert" className="space-y-3">
      <p className="font-sans text-sm text-error">This admin page could not load.</p>
      <p className="font-sans text-sm text-ink-soft">It may be your connection. Press Retry, or reload the page.</p>
      <button type="button" onClick={() => retry()} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
        Retry
      </button>
    </div>
  );
}
