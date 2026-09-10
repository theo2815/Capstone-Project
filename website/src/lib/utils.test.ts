import { afterEach, describe, expect, it, vi } from "vitest";
import { copyToClipboard, safeHttpUrl, triggerDownload } from "./utils";

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals(); document.body.innerHTML = ""; });

describe("copyToClipboard", () => {
  const url = "https://quickpitik.com/quickpitikfree/events/nrw-fun-run-2026-4c711f";

  it("uses the desktop Clipboard API directly in the click gesture", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", { userAgent: "Chrome", clipboard: { writeText } });
    const result = copyToClipboard(url);
    expect(writeText).toHaveBeenCalledWith(url);
    expect(await result).toBe(true);
  });

  it("selects and copies synchronously on iOS before any async rejection, restoring focus", async () => {
    const button = document.createElement("button");
    document.body.appendChild(button);
    button.focus();
    const writeText = vi.fn().mockRejectedValue(new Error("not allowed"));
    vi.stubGlobal("navigator", { userAgent: "iPhone AppleWebKit Safari", clipboard: { writeText } });
    const execCommand = vi.fn(() => {
      const input = document.activeElement as HTMLTextAreaElement;
      expect(input.value).toBe(url);
      expect(input.selectionStart).toBe(0);
      expect(input.selectionEnd).toBe(url.length);
      return true;
    });
    vi.stubGlobal("document", document);
    Object.defineProperty(document, "execCommand", { configurable: true, value: execCommand });
    const result = copyToClipboard(url);
    expect(execCommand).toHaveBeenCalledWith("copy");
    expect(await result).toBe(true);
    expect(writeText).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(button);
    expect(document.querySelector("textarea")).toBeNull();
  });

  it.each([undefined, { writeText: async () => { throw new Error("denied"); } }])(
    "falls back with absent or denied clipboard and reports failure honestly",
    async (clipboard) => {
      vi.stubGlobal("navigator", { userAgent: "Android Chrome", clipboard });
      Object.defineProperty(document, "execCommand", { configurable: true, value: () => { throw new Error("unsupported"); } });
      expect(await copyToClipboard(url)).toBe(false);
      expect(document.querySelector("textarea")).toBeNull();
    },
  );
});

describe("triggerDownload", () => {
  it("submits a large selection in the body without changing or extending the signed URL", () => {
    vi.useFakeTimers();
    const url = "https://api.quickpitik.com/api/v1/photographer-bundles/event?token=signed";
    const ids = Array.from({ length: 3028 }, (_, i) => `photo-${i}`).join(",");
    const submit = vi.spyOn(HTMLFormElement.prototype, "submit").mockImplementation(function (this: HTMLFormElement) {
      expect(this.method).toBe("post");
      expect(this.action).toBe(url);
      expect(new FormData(this).get("photoIds")).toBe(ids);
    });
    triggerDownload(url, { photoIds: ids });
    expect(submit).toHaveBeenCalledOnce();
    expect(document.querySelector("form")).not.toBeNull();
    vi.runAllTimers();
    expect(document.querySelector("form")).toBeNull();
  });
});

describe("safeHttpUrl", () => {
  it("passes http(s) through", () => {
    expect(safeHttpUrl("https://instagram.com/quickpitik")).toBe(
      "https://instagram.com/quickpitik",
    );
    expect(safeHttpUrl("http://localhost:8080/x")).toBe("http://localhost:8080/x");
  });

  it("rejects script-bearing and non-web schemes", () => {
    expect(safeHttpUrl("javascript:alert(1)")).toBeNull();
    expect(safeHttpUrl("JavaScript:alert(1)")).toBeNull();
    expect(safeHttpUrl("data:text/html,<script>1</script>")).toBeNull();
    expect(safeHttpUrl("vbscript:x")).toBeNull();
  });

  it("rejects relative, blank, and malformed input", () => {
    expect(safeHttpUrl("/relative")).toBeNull();
    expect(safeHttpUrl("")).toBeNull();
    expect(safeHttpUrl(null)).toBeNull();
    expect(safeHttpUrl(undefined)).toBeNull();
    expect(safeHttpUrl("not a url")).toBeNull();
  });
});
