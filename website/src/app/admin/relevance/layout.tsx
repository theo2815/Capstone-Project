import type { Metadata } from "next";

export const metadata: Metadata = { title: "Photo Relevance" };

export default function Layout({ children }: { children: React.ReactNode }) {
  return children;
}
