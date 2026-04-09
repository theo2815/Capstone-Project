import Link from "next/link";
import { ROUTES } from "@/lib/constants";
import { Navbar } from "@/components/layout/navbar";
import { Footer } from "@/components/layout/footer";
import {
  Camera,
  Search,
  ShoppingCart,
  ArrowRight,
  Zap,
  Shield,
  Users,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />

      {/* Hero */}
      <section className="relative overflow-hidden bg-charcoal">
        {/* Subtle grid pattern */}
        <div
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,.1) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />

        <div className="relative mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32 lg:px-8 lg:py-40">
          <div className="max-w-2xl">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-steel-blue/30 bg-steel-blue/10 px-4 py-1.5">
              <Zap className="h-3.5 w-3.5 text-teal-light" />
              <span className="text-xs font-medium text-cool-gray">
                AI-Powered Photo Search
              </span>
            </div>

            <h1 className="text-4xl font-bold tracking-tight text-white sm:text-5xl lg:text-6xl">
              Find your race photos
              <span className="block text-teal-light">in seconds.</span>
            </h1>

            <p className="mt-6 max-w-lg text-lg leading-relaxed text-cool-gray">
              Upload a selfie or enter your bib number — our AI finds every
              photo of you across the entire marathon event.
            </p>

            <div className="mt-10 flex flex-col gap-3 sm:flex-row sm:gap-4">
              <Link
                href={ROUTES.EVENTS}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-3 text-base font-medium text-white hover:bg-primary-hover transition-colors"
              >
                Browse Events
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link
                href={ROUTES.REGISTER}
                className="inline-flex items-center justify-center rounded-lg border border-steel-blue/40 px-6 py-3 text-base font-medium text-cool-gray hover:border-cool-gray hover:text-white transition-colors"
              >
                Sign Up Free
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="scroll-mt-20 bg-surface py-20 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-charcoal sm:text-4xl">
              How It Works
            </h2>
            <p className="mx-auto mt-3 max-w-xl text-charcoal-light">
              From the finish line to your photo — three simple steps.
            </p>
          </div>

          <div className="mt-16 grid gap-8 sm:grid-cols-3">
            {[
              {
                icon: Camera,
                step: "01",
                title: "Photographers Upload",
                description:
                  "Event photographers upload race photos. Our AI automatically detects faces and reads bib numbers.",
              },
              {
                icon: Search,
                step: "02",
                title: "Runners Search",
                description:
                  "Search by selfie or bib number to instantly find all your race photos across the event.",
              },
              {
                icon: ShoppingCart,
                step: "03",
                title: "Purchase & Download",
                description:
                  "Preview watermarked photos, add to cart, and download high-resolution originals after purchase.",
              },
            ].map((item) => (
              <div
                key={item.step}
                className="group relative rounded-2xl border border-border bg-background p-8 transition-shadow hover:shadow-lg"
              >
                <span className="text-5xl font-bold text-cool-gray-light">
                  {item.step}
                </span>
                <div className="mt-4 flex h-12 w-12 items-center justify-center rounded-xl bg-teal/10">
                  <item.icon className="h-6 w-6 text-primary" />
                </div>
                <h3 className="mt-4 text-lg font-semibold text-charcoal">
                  {item.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-charcoal-light">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features / Why Quick Pitik */}
      <section className="bg-background py-20 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h2 className="text-3xl font-bold text-charcoal sm:text-4xl">
              Why Quick Pitik?
            </h2>
            <p className="mx-auto mt-3 max-w-xl text-charcoal-light">
              Built for runners and photographers in the Philippines.
            </p>
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {[
              {
                icon: Zap,
                title: "AI-Powered Search",
                description:
                  "Face recognition and OCR find your photos faster than scrolling through thousands of images.",
              },
              {
                icon: Shield,
                title: "Watermark Protection",
                description:
                  "Photos are watermarked during preview. High-res originals are only available after purchase.",
              },
              {
                icon: Users,
                title: "For Photographers",
                description:
                  "Upload your event photos and earn from every download. We handle the AI tagging for you.",
              },
            ].map((feature) => (
              <div
                key={feature.title}
                className="rounded-2xl border border-border p-6"
              >
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-steel-blue/10">
                  <feature.icon className="h-5 w-5 text-steel-blue" />
                </div>
                <h3 className="mt-4 font-semibold text-charcoal">
                  {feature.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-charcoal-light">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="bg-primary py-16 sm:py-20">
        <div className="mx-auto max-w-7xl px-4 text-center sm:px-6 lg:px-8">
          <h2 className="text-3xl font-bold text-white sm:text-4xl">
            Ready to find your photos?
          </h2>
          <p className="mx-auto mt-3 max-w-md text-teal-light/80">
            Browse upcoming events and search for your race photos today.
          </p>
          <div className="mt-8 flex flex-col items-center justify-center gap-3 sm:flex-row sm:gap-4">
            <Link
              href={ROUTES.EVENTS}
              className="inline-flex items-center gap-2 rounded-lg bg-white px-6 py-3 text-base font-medium text-primary hover:bg-warm-white transition-colors"
            >
              View Events
              <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              href={ROUTES.REGISTER}
              className="inline-flex items-center rounded-lg border border-white/30 px-6 py-3 text-base font-medium text-white hover:border-white hover:bg-white/10 transition-colors"
            >
              Create Account
            </Link>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}
