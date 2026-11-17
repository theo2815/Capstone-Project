import { act } from "@testing-library/react";
import { hydrateRoot } from "react-dom/client";
import { renderToString } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BibNotice, FaceNotice } from "@/components/legal/face-notice";
import { useBibConsentStore, useFaceConsentStore } from "@/store/face-consent-store";

// BT-1c: the server cannot read this device's saved tick, so it must not draw
// the notice (a device that already ticked would see it flash before the JS loads).
const cases = [
  { name: "FaceNotice", Notice: FaceNotice, store: useFaceConsentStore, box: /I agree to face matching/ },
  { name: "BibNotice", Notice: BibNotice, store: useBibConsentStore, box: /This is my bib number/ },
];

afterEach(() => {
  useFaceConsentStore.setState({ acceptedVersion: null });
  useBibConsentStore.setState({ acceptedVersion: null });
  document.body.innerHTML = "";
});

async function hydrate(html: string, ui: React.ReactNode) {
  const container = document.createElement("div");
  container.innerHTML = html;
  document.body.appendChild(container);
  const onRecoverableError = vi.fn();
  await act(async () => {
    hydrateRoot(container, ui, { onRecoverableError });
  });
  return { container, onRecoverableError };
}

describe.each(cases)("$name", ({ Notice, store, box }) => {
  it("is not in the server-drawn HTML", () => {
    expect(renderToString(<Notice />)).toBe("");
  });

  it("stays hidden after hydration on a device that already ticked, with no mismatch", async () => {
    const html = renderToString(<Notice />);
    store.getState().accept();
    const { container, onRecoverableError } = await hydrate(html, <Notice />);
    expect(container.textContent).toBe("");
    expect(onRecoverableError).not.toHaveBeenCalled();
  });

  it("appears after hydration on a device that has not ticked", async () => {
    const { container, onRecoverableError } = await hydrate(renderToString(<Notice />), <Notice />);
    expect(container.textContent).toMatch(box);
    expect(onRecoverableError).not.toHaveBeenCalled();
  });
});
