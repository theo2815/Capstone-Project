// Detects in-app / embedded browsers (Facebook, Messenger, Instagram, TikTok,
// Line, Twitter, generic Android WebView) where Google deliberately blocks
// OAuth with `disallowed_useragent`. The GIS button dead-ends there on a
// "browser not secure" screen while email/password still works — the confirmed
// root cause of the NRW Fun Run "Google works for some runners, not others"
// report. We use this to hide the button and steer runners to email or a real
// browser instead of walking them into Google's dead-end.
const IN_APP_BROWSER_RE =
  /\bFBAN\b|\bFBAV\b|FB_IAB|Messenger|Instagram|\bLine\b|Twitter|musical_ly|BytedanceWebview|TikTok|;\s?wv\b/i;

const IOS_DEVICE_RE = /\b(?:iPhone|iPad|iPod)\b/i;
const IOS_BROWSER_RE = /\b(?:Safari|CriOS|FxiOS|EdgiOS|OPiOS)\b/i;

export function isInAppBrowser(
  ua: string | undefined = typeof navigator === "undefined"
    ? undefined
    : navigator.userAgent,
): boolean {
  if (!ua) return false;
  return (
    IN_APP_BROWSER_RE.test(ua) ||
    (IOS_DEVICE_RE.test(ua) &&
      /AppleWebKit/i.test(ua) &&
      !IOS_BROWSER_RE.test(ua))
  );
}

// True on iPhone / iPad / iPod. Used to prefer the native share sheet, whose
// "Save Image" is the only route into the iOS camera roll — a plain download on
// iOS Safari lands in the Files app instead. (iPadOS Safari's desktop-mode UA
// drops the "iPad" token and reads as a Mac; that falls back to the plain
// download, which is acceptable.)
export function isIOS(
  ua: string | undefined = typeof navigator === "undefined"
    ? undefined
    : navigator.userAgent,
): boolean {
  return ua ? IOS_DEVICE_RE.test(ua) : false;
}
