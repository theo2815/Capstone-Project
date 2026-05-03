import type { ReactNode } from "react";
import { SiteHeader } from "@/components/layout/site-header";
import { AuthEditorialAside } from "@/components/auth/auth-editorial-aside";

interface AuthShellProps {
  rightLink: { label: string; href: string };
  children: ReactNode;
}

export function AuthShell({ rightLink, children }: AuthShellProps) {
  return (
    <>
      <SiteHeader rightLink={rightLink} />
      <main className="flex-1 px-6 md:px-10 py-16 md:py-24">
        <div className="max-w-7xl mx-auto w-full grid md:grid-cols-2 gap-12 md:gap-20 items-start md:items-center">
          <div className="w-full max-w-[380px] mx-auto md:mx-0">
            {children}
          </div>
          <AuthEditorialAside />
        </div>
      </main>
    </>
  );
}
