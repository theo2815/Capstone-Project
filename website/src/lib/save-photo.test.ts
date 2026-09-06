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

import { dataUrlToFile, savePhoto, saveQrCode, sharePhotos } from "./save-photo";

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

const URLS = [
  "https://api.quickpitik.com/api/v1/orders/o/download-bundle?token=t&photo=1",
  "https://api.quickpitik.com/api/v1/orders/o/download-bundle?token=t&photo=2",
];

// A minimal Response stand-in — enough of the shape save-photo actually reads.
const okRes = () => ({
  ok: true,
  status: 200,
  blob: async () => new Blob(["x"], { type: "image/jpeg" }),
  headers: { get: () => null },
});

function stubFetch(impl: (url: string) => Promise<unknown>) {
  vi.stubGlobal("fetch", vi.fn((input: string) => impl(String(input))));
}

function stubNavigatorShare(canShare: boolean, share: () => Promise<void> = async () => {}) {
  vi.stubGlobal("navigator", {
    userAgent: "test",
    canShare: vi.fn(() => canShare),
    share: vi.fn(share),
  });
}

const shareArg = () =>
  (navigator.share as ReturnType<typeof vi.fn>).mock.calls[0][0] as {
    files: File[];
  };

describe("sharePhotos", () => {
  it("is unsupported when the browser has no Web Share API", async () => {
    vi.stubGlobal("navigator", { userAgent: "test" });
    expect((await sharePhotos(URLS)).status).toBe("unsupported");
  });

  it("is unsupported when canShare rejects the files", async () => {
    stubFetch(async () => okRes());
    stubNavigatorShare(false);
    const res = await sharePhotos(URLS);
    expect(res.status).toBe("unsupported");
    expect(navigator.share).not.toHaveBeenCalled();
  });

  it("shares every fetched photo as one batch and reports progress", async () => {
    stubFetch(async () => okRes());
    stubNavigatorShare(true);
    const onProgress = vi.fn();
    const res = await sharePhotos(URLS, { onProgress });
    expect(res.status).toBe("shared");
    expect(res.sharedUrls).toEqual(expect.arrayContaining(URLS));
    expect(res.failedUrls).toEqual([]);
    expect(shareArg().files).toHaveLength(2);
    expect(shareArg().files[0]).toBeInstanceOf(File);
    expect(onProgress).toHaveBeenLastCalledWith({ done: 2, total: 2 });
  });

  it("shares the reachable photos and reports the ones that failed to fetch", async () => {
    stubFetch(async (url) =>
      url.endsWith("photo=2")
        ? { ok: false, status: 500, headers: { get: () => null } }
        : okRes(),
    );
    stubNavigatorShare(true);
    const res = await sharePhotos(URLS);
    expect(res.status).toBe("shared");
    expect(res.sharedUrls).toEqual([URLS[0]]);
    expect(res.failedUrls).toEqual([URLS[1]]);
    expect(shareArg().files).toHaveLength(1);
  });

  it("retries a transient fetch failure before giving up", async () => {
    let calls = 0;
    stubFetch(async () => {
      calls += 1;
      if (calls === 1) throw new TypeError("network");
      return okRes();
    });
    stubNavigatorShare(true);
    const res = await sharePhotos([URLS[0]]);
    expect(res.status).toBe("shared");
    expect(res.sharedUrls).toEqual([URLS[0]]);
    expect(calls).toBeGreaterThanOrEqual(2);
  });

  it("treats a dismissed sheet (AbortError) as shared", async () => {
    stubFetch(async () => okRes());
    stubNavigatorShare(true, async () => {
      throw new DOMException("cancelled", "AbortError");
    });
    expect((await sharePhotos(URLS)).status).toBe("shared");
  });

  it("returns error when the share throws (the gesture lapsed)", async () => {
    stubFetch(async () => okRes());
    stubNavigatorShare(true, async () => {
      throw new DOMException("not allowed", "NotAllowedError");
    });
    expect((await sharePhotos(URLS)).status).toBe("error");
  });

  it("returns error when nothing could be fetched", async () => {
    stubFetch(async () => ({ ok: false, status: 500, headers: { get: () => null } }));
    stubNavigatorShare(true);
    const res = await sharePhotos([URLS[0]]);
    expect(res.status).toBe("error");
    expect(res.failedUrls).toEqual([URLS[0]]);
    expect(navigator.share).not.toHaveBeenCalled();
  });
});

// The inline QR is a data: URL, not a signed http URL — decoded locally, no fetch.
const QR_DATA_URL = "data:image/png;base64,cXJwaA=="; // base64 of "qrph" (4 bytes)
const QR_NAME = "quickpitik-qrph-order.png";

describe("saveQrCode", () => {
  it("shares the decoded QR image in an in-app browser", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    stubNavigatorShare(true);
    expect(await saveQrCode(QR_DATA_URL, QR_NAME)).toBe("shared");
    expect(navigator.share).toHaveBeenCalledOnce();
    const file = shareArg().files[0];
    expect(file).toBeInstanceOf(File);
    expect(file.type).toBe("image/png");
    expect(showToastMock).not.toHaveBeenCalled();
  });

  it("hints how to save when an in-app webview can't share the QR", async () => {
    isInAppMock.mockReturnValue(true);
    isIOSMock.mockReturnValue(false);
    stubNavigatorShare(false);
    expect(await saveQrCode(QR_DATA_URL, QR_NAME)).toBe("hint");
    expect(navigator.share).not.toHaveBeenCalled();
    expect(showToastMock).toHaveBeenCalledOnce();
  });

  it("downloads (never hints) on iOS Safari without file sharing", async () => {
    isInAppMock.mockReturnValue(false);
    isIOSMock.mockReturnValue(true);
    stubNavigatorShare(false);
    expect(await saveQrCode(QR_DATA_URL, QR_NAME)).toBe("downloaded");
    expect(showToastMock).not.toHaveBeenCalled();
  });
});

describe("dataUrlToFile", () => {
  it("decodes a base64 data URL into a typed File", () => {
    const file = dataUrlToFile(QR_DATA_URL, "q.png");
    expect(file).toBeInstanceOf(File);
    expect(file?.type).toBe("image/png");
    expect(file?.size).toBe(4); // "qrph"
  });

  it("returns null on a non-data URL", () => {
    expect(dataUrlToFile("https://example.com/x.png", "q.png")).toBeNull();
  });
});
