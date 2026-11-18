import type { Metadata } from "next";

// A string title here would clear the admin template for the pages below, so repeat it.
export const metadata: Metadata = { title: { default: "Photo removals", template: "%s · Admin | QuickPitik" } };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
