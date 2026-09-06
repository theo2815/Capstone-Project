import { afterEach, describe, expect, it, vi } from "vitest";

// Shared spies, hoisted so the vi.mock factories below may reference them.
const { triggerDownloadMock, showToastMock, isInAppMock, isIOSMock } = vi.hoisted(
  () => ({
    triggerDownloadMock: vi.fn(),
    showToastMock: vi.fn(),
    isInAppMock: vi.fn<(ua?: string) => boolean>(),
    isIOSMock: vi.fn<(ua?: string) => boolean>(),
  }),
);

vi.mock("@/lib/in-app-browser", () => ({
  isInAppBrowser: isInAppMock,
  isIOS: isIOSMock,
}));
vi.mock("@/lib/utils", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/utils")>();
  return { ...actual, triggerDownload: triggerDownloadMock };
});
vi.mock("@/store/toast-store", () => ({
  useToastStore: { getState: () => ({ showToast: showToastMock }) },
}));

import { savePhoto } from "./save-photo";

const URL = "https://api.quickpitik.com/api/v1/events/e/photos/p/download";

// jsdom has no Web Share; install a fake navigator with the two methods.
function stubShare(canShare: boolean, share: () => Promise<void> = async () => {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      blob: async () => new Blob(["x"], { type: "image/jpeg" }),
    }),
  );
  vi.stubGlobal("navigator", {
    userAgent: "test",
    canShare: vi.fn(() => canShare),
    share: vi.fn(share),
  });
}

afterEach(() => {
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

describe("savePhoto", () => {
  it("downloads via the anchor on desktop / Android (not in-app, not iOS)", async () => {
    isInAppMock.mockReturnValue(false);
    isIOSMock.mockReturnValue(false);
    expect(await savePhoto(URL)).toBe("downloaded");
    expect(triggerDownloadMock).toHaveBeenCalledWith(URL);
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("shares the file on iOS Safari so it can reach Photos", async () => {
    isInAppMock.mockReturnValue(false);
    isIOSMock.mockReturnValue(true);
    stubShare(true);
    expect(await savePhoto(URL)).toBe("shared");
    expect(navigator.share).toHaveBeenCalledOnce();
    expect(triggerDownloadMock).not.toHaveBeenCalled();
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("falls back to a plain download on old iOS without file sharing", async () => {
    isInAppMock.mockReturnValue(false);
    isIOSMock.mockReturnValue(true);
    stubShare(false);
    expect(await savePhoto(URL)).toBe("downloaded");
    expect(triggerDownloadMock).toHaveBeenCalledWith(URL);
    expect(showToastMock).not.toHaveBeenCalled(); // never the in-app hint in a real browser
  });

  it("shares the file in an in-app browser that supports file sharing", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    stubShare(true);
    expect(await savePhoto(URL)).toBe("shared");
    expect(navigator.share).toHaveBeenCalledOnce();
    const arg = (navigator.share as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(arg.files[0]).toBeInstanceOf(File);
    expect(triggerDownloadMock).not.toHaveBeenCalled();
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("treats a dismissed share sheet (AbortError) as done, no hint", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    stubShare(true, async () => {
      throw new DOMException("cancelled", "AbortError");
    });
    expect(await savePhoto(URL)).toBe("shared");
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("hints how to save when an in-app webview can't share files", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    stubShare(false);
    expect(await savePhoto(URL)).toBe("hint");
    expect(navigator.share).not.toHaveBeenCalled();
    expect(showToastMock).toHaveBeenCalledOnce();
  });

  it("hints when an in-app webview has no Web Share API at all", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    vi.stubGlobal("navigator", { userAgent: "test" });
    expect(await savePhoto(URL)).toBe("hint");
    expect(showToastMock).toHaveBeenCalledOnce();
  });
});
