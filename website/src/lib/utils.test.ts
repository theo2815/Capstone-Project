import { describe, expect, it } from "vitest";
import { safeHttpUrl } from "./utils";

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
