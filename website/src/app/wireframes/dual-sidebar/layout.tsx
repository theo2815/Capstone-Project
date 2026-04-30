import type { ReactNode } from "react";
import { Newsreader, Source_Serif_4, Caveat } from "next/font/google";

const newsreader = Newsreader({
  variable: "--font-notebook-display",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  style: ["normal", "italic"],
  display: "swap",
});

const sourceSerif = Source_Serif_4({
  variable: "--font-notebook-body",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  style: ["normal", "italic"],
  display: "swap",
});

const caveat = Caveat({
  variable: "--font-notebook-hand",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  display: "swap",
});

export default function DualSidebarLayout({
  children,
}: {
  children: ReactNode;
}) {
  return (
    <div
      className={`${newsreader.variable} ${sourceSerif.variable} ${caveat.variable}`}
    >
      {children}
    </div>
  );
}
