import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { LEGAL_VERSION } from "@/lib/legal/version";

// The terms version has two homes: LEGAL_VERSION here and the backend's
// app.legal.current-version (stamped on new sign-ups). This keeps them equal.
// Resolved from website/ (vitest's root): jsdom gives import.meta.url a non-file scheme.
const BACKEND_YML = resolve(process.cwd(), "../backend/src/main/resources/application.yml");

describe("legal version", () => {
  it("matches the backend's app.legal.current-version", () => {
    const yml = readFileSync(BACKEND_YML, "utf8");
    const backend = yml.match(/^\s+current-version:\s*"([^"]+)"/m)?.[1];
    expect(backend).toBe(LEGAL_VERSION);
  });
});
