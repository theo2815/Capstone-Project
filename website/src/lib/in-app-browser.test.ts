import { describe, expect, it } from "vitest";
import { isInAppBrowser } from "./in-app-browser";

// Real-world UA shapes. Sources: the app strings each embedded browser appends.
const FB_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 [FBAN/FBIOS;FBAV/450.0.0.0.0;]";
const MESSENGER_ANDROID =
  "Mozilla/5.0 (Linux; Android 12; SM-A125F Build/SP1A.210812.016; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0.0.0 Mobile Safari/537.36 [FB_IAB/Orca-Android;FBAV/430.0.0.0;]";
const INSTAGRAM =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Instagram 300.0.0.0.0 (iPhone14,3; iOS 17_0; en_US)";
const ANDROID_WEBVIEW =
  "Mozilla/5.0 (Linux; Android 13; Pixel 7 Build/TQ2A.230505.002; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/120.0.0.0 Mobile Safari/537.36";

const CHROME_ANDROID =
  "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36";
const SAFARI_IOS =
  "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1";

describe("isInAppBrowser", () => {
  it("flags embedded webviews that block Google OAuth", () => {
    for (const ua of [FB_IOS, MESSENGER_ANDROID, INSTAGRAM, ANDROID_WEBVIEW]) {
      expect(isInAppBrowser(ua)).toBe(true);
    }
  });

  it("passes real mobile browsers where OAuth works", () => {
    for (const ua of [CHROME_ANDROID, SAFARI_IOS]) {
      expect(isInAppBrowser(ua)).toBe(false);
    }
  });

  it("is false when the UA is unknown (SSR)", () => {
    expect(isInAppBrowser(undefined)).toBe(false);
  });
});
