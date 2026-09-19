import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

class MemoryStorage implements Storage {
  private readonly entries = new Map<string, string>();

  get length() {
    return this.entries.size;
  }

  clear() {
    this.entries.clear();
  }

  getItem(key: string) {
    return this.entries.get(String(key)) ?? null;
  }

  key(index: number) {
    return Array.from(this.entries.keys())[index] ?? null;
  }

  removeItem(key: string) {
    this.entries.delete(String(key));
  }

  setItem(key: string, value: string) {
    this.entries.set(String(key), String(value));
  }
}

// Node 26 can expose an undefined native localStorage without a storage file.
// Use the same browser-compatible storage for both Zustand and window callers.
const storage = new MemoryStorage();
Object.defineProperty(globalThis, "localStorage", { configurable: true, value: storage });
Object.defineProperty(window, "localStorage", { configurable: true, value: storage });

// RTL only auto-cleans with `globals: true`; without this, a second `render`
// in the same file finds the first test's DOM still mounted.
afterEach(() => {
  cleanup();
  storage.clear();
});
