import type { Metadata, Viewport } from "next";
import { Anton, Archivo, Geist_Mono } from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";

// Finish Line type system (2026-08-25 overhaul):
//   Anton      → hero display + big race numbers (condensed, uppercase, max impact)
//   Archivo    → headings + UI + body (athletic grotesque, weights 400–900)
//   Geist Mono → bibs / times / prices / stats (race-clock feel)
const anton = Anton({
  variable: "--font-anton",
  subsets: ["latin"],
  weight: ["400"],
  display: "swap",
});

const archivo = Archivo({
  variable: "--font-archivo",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800", "900"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL ?? "https://quickpitik.com";

export const metadata: Metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: "QuickPitik — Race Photos, Delivered in Minutes",
    template: "%s | QuickPitik",
  },
  description:
    "Marathon photos delivered minutes after the finish line. Find yours by face or bib in seconds.",
  openGraph: {
    type: "website",
    siteName: "QuickPitik",
    images: ["/brand/quickpitik-logo-transparent.png"],
  },
  twitter: { card: "summary_large_image" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#F8F5EE",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${anton.variable} ${archivo.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
