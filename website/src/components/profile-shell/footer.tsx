import { FinishTape } from "@/components/layout/footer";
import { LegalLinks } from "@/components/legal/legal-links";
import { Kicker } from "@/components/ui/kicker";

// Slim dark bar shared across signed-in profile-shell pages. Mounted at the
// bottom of /profile, /orders, /account (with its sub-pages) and the public
// /[handle]/events/[slug] page.
export function ProfileShellFooter() {
  return (
    <footer className="mt-12 bg-pine">
      <FinishTape />
      <div className="flex flex-col items-start justify-between gap-3 px-6 py-5 md:flex-row md:items-center md:px-10">
        <Kicker tnum className="text-paper/70">
          {`© ${new Date().getFullYear()} QuickPitik`}
        </Kicker>
        <LegalLinks onDark />
        <Kicker className="text-paper/70">Cebu, Philippines</Kicker>
      </div>
    </footer>
  );
}
