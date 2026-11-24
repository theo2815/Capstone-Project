import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Slab } from "./slab";

afterEach(cleanup);

// The non-admin pages (/dashboard, /account, /orders, /profile) keep this exact title look.
const LABEL_TITLE =
  "font-mono font-normal uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-ink min-w-0 md:shrink-0";

describe("Slab title style", () => {
  it("keeps the mono uppercase label by default", () => {
    render(<Slab number="01" title="Key metrics">body</Slab>);
    const h2 = screen.getByRole("heading", { level: 2, name: "Key metrics" });
    expect(h2.className).toBe(LABEL_TITLE);
  });

  it("reads as a sentence-case heading with titleStyle=heading", () => {
    render(<Slab number="01" title="Key metrics" caption="Totals right now." titleStyle="heading">body</Slab>);
    const h2 = screen.getByRole("heading", { level: 2, name: "Key metrics" });
    expect(h2.className).toContain("normal-case");
    expect(h2.className).toContain("font-semibold");
    expect(h2.className).not.toMatch(/(^|\s)uppercase(\s|$)/);
    expect(h2.className).not.toContain("font-mono");
    // The number and the caption stay mono beside it.
    expect(screen.getByText("01").className).toContain("font-mono");
    expect(screen.getByText("Totals right now.").className).toContain("font-mono");
  });
});
