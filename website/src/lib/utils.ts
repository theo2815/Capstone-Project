import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(date: string): string {
  return new Intl.DateTimeFormat("en-PH", {
    year: "numeric",
    month: "long",
    day: "numeric",
  }).format(new Date(date));
}

export function formatPrice(amount: number): string {
  return new Intl.NumberFormat("en-PH", {
    style: "currency",
    currency: "PHP",
  }).format(amount);
}

// Only http(s) may reach an href / location sink. `javascript:` and `data:`
// URLs execute under this app's 'unsafe-inline' script-src, so every
// user- or backend-supplied URL that becomes a navigation goes through here.
export function safeHttpUrl(u: string | null | undefined): string | null {
  try {
    const p = new URL(u ?? "");
    return p.protocol === "https:" || p.protocol === "http:" ? p.href : null;
  } catch {
    return null;
  }
}

// Guarded clipboard write. Resolves true/false (never throws) so callers can
// toast an honest "couldn't copy" instead of a silent no-op or a false
// success. Tries the async Clipboard API first, then falls back to the legacy
// execCommand("copy") path for contexts where navigator.clipboard is absent or
// rejects — insecure origins and in-app browsers (Facebook/Instagram WebView),
// which is exactly where photographers open share links.
export async function copyToClipboard(text: string): Promise<boolean> {
  if (typeof navigator !== "undefined" && navigator.clipboard) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // fall through to the legacy path
    }
  }
  return legacyCopy(text);
}

function legacyCopy(text: string): boolean {
  if (typeof document === "undefined") return false;
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    // Keep it off-screen and non-disruptive: no scroll jump, no focus flash.
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-9999px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    ta.remove();
    return ok;
  } catch {
    return false;
  }
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .replace(/\s+/g, "-")
    .trim();
}

// Short order reference — the same 8-char form the receipt email prints, so
// a runner can match what they see on screen against their inbox.
export function formatOrderRef(orderId: string): string {
  return orderId.slice(0, 8).toUpperCase();
}

export function safeUUID(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  // Fallback UUID v4 generator for non-secure HTTP contexts
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// Programmatic save of a signed attachment URL — the same anchor-click the
// orders page and the photographer library use, shared because free-event
// tiles, both lightboxes and the share page all need it.
export function triggerDownload(url: string): void {
  const href = safeHttpUrl(url);
  if (!href) return;
  const a = document.createElement("a");
  a.href = href;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
