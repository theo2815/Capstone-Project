import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import { isIOS, isInAppBrowser } from "@/lib/in-app-browser";

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
// success. Desktop prefers the async Clipboard API; iOS/webviews try a
// synchronous copy first to retain the tap. The legacy path also covers
// unavailable/denied clipboard access, including insecure development origins.
export async function copyToClipboard(text: string): Promise<boolean> {
  // WebKit/webviews can lose the tap's activation while writeText rejects.
  // Try their synchronous fallback while still inside the click handler.
  if ((isIOS() || isInAppBrowser()) && legacyCopy(text)) return true;
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
  const focused = document.activeElement;
  const selection = document.getSelection();
  const ranges = selection
    ? Array.from({ length: selection.rangeCount }, (_, i) => selection.getRangeAt(i).cloneRange())
    : [];
  const ta = document.createElement("textarea");
  try {
    ta.value = text;
    // Keep it focusable on iOS without a scroll jump, zoom, or focus flash.
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "0";
    ta.style.left = "0";
    ta.style.fontSize = "16px";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus({ preventScroll: true });
    ta.select();
    ta.setSelectionRange(0, text.length);
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    ta.remove();
    if (focused instanceof HTMLElement) focused.focus({ preventScroll: true });
    if (selection) {
      selection.removeAllRanges();
      ranges.forEach((range) => selection.addRange(range));
    }
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
export function triggerDownload(url: string, fields?: Record<string, string>): void {
  const href = safeHttpUrl(url);
  if (!href) return;
  if (fields) {
    const form = document.createElement("form");
    form.method = "POST";
    form.action = href;
    form.hidden = true;
    for (const [name, value] of Object.entries(fields)) {
      const input = document.createElement("input");
      input.type = "hidden";
      input.name = name;
      input.value = value;
      form.appendChild(input);
    }
    document.body.appendChild(form);
    form.submit();
    // Submission queues navigation; detaching synchronously cancels it in Chrome.
    setTimeout(() => form.remove(), 1000);
    return;
  }
  const a = document.createElement("a");
  a.href = href;
  a.download = "";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
