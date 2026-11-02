import { describe, expect, it } from "vitest";
import { assertPublishable } from "./publishable";

describe("assertPublishable", () => {
  it("throws on production when a placeholder is left", () => {
    expect(() =>
      assertPublishable(["Last updated [ASK OWNER: date]"], "1.0", "production"),
    ).toThrow(/\[ASK OWNER: date\]/);
  });

  it("throws on production when the version is still a draft", () => {
    expect(() => assertPublishable(["Clean text."], "1.0-draft", "production")).toThrow(
      /1\.0-draft/,
    );
  });

  it("lets a preview build show the draft", () => {
    expect(() =>
      assertPublishable(["[ASK OWNER: date]"], "1.0-draft", "preview"),
    ).not.toThrow();
  });

  it("passes on production when the text is final", () => {
    expect(() => assertPublishable(["Clean text."], "1.0", "production")).not.toThrow();
  });
});
