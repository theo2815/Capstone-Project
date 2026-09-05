// Detects in-app / embedded browsers (Facebook, Messenger, Instagram, TikTok,
// Line, Twitter, generic Android WebView) where Google deliberately blocks
// OAuth with `disallowed_useragent`. The GIS button dead-ends there on a
// "browser not secure" screen while email/password still works — the confirmed
// root cause of the NRW Fun Run "Google works for some runners, not others"
// report. We use this to hide the button and steer runners to email or a real
// browser instead of walking them into Google's dead-end.
const IN_APP_BROWSER_RE =
  /\bFBAN\b|\bFBAV\b|FB_IAB|Messenger|Instagram|\bLine\b|Twitter|musical_ly|BytedanceWebview|TikTok|;\s?wv\b/i;

export function isInAppBrowser(
  ua: string | undefined = typeof navigator === "undefined"
    ? undefined
    : navigator.userAgent,
): boolean {
  return ua != null && IN_APP_BROWSER_RE.test(ua);
}
