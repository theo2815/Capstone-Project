import { notFound } from "next/navigation";

// Dev-only SRS wireframes (13 static mock pages under /wireframes/srs). They
// render no data, but shipping them in production is free reconnaissance on
// internal module naming — the whole segment 404s outside development.
export default function WireframesLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  if (process.env.NODE_ENV === "production") notFound();
  return children;
}
