import type { Metadata } from "next";
import {
  Bricolage_Grotesque,
  Funnel_Sans,
  Geist_Mono,
} from "next/font/google";
import { Providers } from "./providers";
import "./globals.css";

const bricolage = Bricolage_Grotesque({
  variable: "--font-bricolage",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "800"],
  display: "swap",
});

const funnel = Funnel_Sans({
  variable: "--font-funnel",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "QuickPitik — Race Photos, Delivered in Minutes",
  description:
    "Marathon photos delivered minutes after the finish line. Find yours by face or bib in seconds.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${bricolage.variable} ${funnel.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
