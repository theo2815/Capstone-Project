// Pure helpers for the BatchMyPhotos connect hand-off
// (docs/desktop-upload-integration-guide.md § 1). `state` and `challenge`
// arrive from the desktop as base64url of 32 random bytes — 43 characters.
// Anything else is not a link the app produced.
const DESKTOP_CONNECT_PARAM = /^[A-Za-z0-9_-]{43}$/;

export function isDesktopConnectParam(value: string | null | undefined): value is string {
  return typeof value === "string" && DESKTOP_CONNECT_PARAM.test(value);
}

export function buildDesktopCallbackUrl(code: string, state: string): string {
  return `batchmyphotos://quickpitik/callback?code=${encodeURIComponent(code)}&state=${encodeURIComponent(state)}`;
}

// Kept separate so tests can observe the navigation: jsdom cannot follow a
// custom scheme, and the real browser hands the URL to the OS protocol handler.
export function navigateTo(url: string): void {
  window.location.assign(url);
}
