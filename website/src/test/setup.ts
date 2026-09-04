import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

// RTL only auto-cleans with `globals: true`; without this, a second `render`
// in the same file finds the first test's DOM still mounted.
afterEach(cleanup);
