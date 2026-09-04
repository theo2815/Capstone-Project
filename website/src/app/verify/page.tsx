import type { Metadata } from "next";
import { SiteHeader } from "@/components/layout/site-header";
import { Footer } from "@/components/layout/footer";
import { VerifyForm } from "./verify-form";

export const metadata: Metadata = {
  title: "Verify a photo | QuickPitik",
  description:
    "Check whether an image is a QuickPitik race photo and which photographer took it — even after a screenshot, crop, or re-save.",
};

export default function VerifyPage() {
  return (
    <main className="bg-bone text-ink relative">
      <SiteHeader />
      <VerifyForm />
      <Footer />
    </main>
  );
}
