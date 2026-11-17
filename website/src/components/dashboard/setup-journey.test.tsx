import { render, screen } from "@testing-library/react";
import { beforeEach, expect, it, vi } from "vitest";
import type { UploadGateState } from "@/hooks/use-can-upload";
import { SetupJourney } from "./setup-journey";

const mocks = vi.hoisted(() => ({ gate: { kind: "pending" } as UploadGateState }));
vi.mock("@/hooks/use-can-upload", () => ({ useCanUpload: () => mocks.gate }));
vi.mock("@/hooks/use-photographer-data", () => ({ usePhotographerEvents: () => [] }));

beforeEach(() => {
  mocks.gate = { kind: "pending" };
});

it("a held profile is told an admin is reviewing it, with no auto-approval promise", () => {
  mocks.gate = { kind: "pending", awaitingAdmin: true };
  render(<SetupJourney />);
  expect(screen.getByText("An admin is reviewing your profile.")).toBeInTheDocument();
  expect(screen.queryByText(/goes live as soon as/)).toBeNull();
  expect(screen.queryByText(/submit them for review/)).toBeNull();
});

it("a plain pending profile keeps today's text", () => {
  render(<SetupJourney />);
  expect(screen.getByText("Submit your settings. Your studio goes live as soon as your identity check passes.")).toBeInTheDocument();
  expect(screen.getByText("Your settings look good — submit them for review when you're ready.")).toBeInTheDocument();
  expect(screen.queryByText("An admin is reviewing your profile.")).toBeNull();
});
