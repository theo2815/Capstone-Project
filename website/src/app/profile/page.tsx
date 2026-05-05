"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { ProtectedRoute } from "@/components/auth/protected-route";
import { SiteHeader } from "@/components/layout/site-header";
import { AvatarDisc } from "@/components/account/avatar-disc";
import { SelfieLibrary } from "@/components/profile/selfie-library";
import { useAuth } from "@/hooks/use-auth";
import { useSavedEventsStore } from "@/store/saved-events-store";
import { useOrdersStore } from "@/store/orders-store";
import { useToast } from "@/hooks/use-toast";
import { EVENT_CATALOG, MOCK_USER_PHOTOS_FOUND } from "@/lib/event-catalog";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";
import type { Role, User } from "@/types/user";

type RaceState = "upcoming" | "live" | "open" | "past";

interface RaceLogEntry {
  id: string;
  slug: string;
  name: string;
  date: string;
  state: RaceState;
  photosFound: number;
  photosBought: number;
  isSaved: boolean;
}

const JUMP_SECTIONS: ReadonlyArray<{ id: string; label: string }> = [
  { id: "selfies", label: "Selfie library" },
  { id: "race-log", label: "Race log" },
];

export default function ProfilePage() {
  return (
    <ProtectedRoute>
      <ProfileBody />
    </ProtectedRoute>
  );
}

function ProfileBody() {
  const { user } = useAuth();
  if (!user) return null;

  return (
    <main className="bg-bone text-ink min-h-screen flex flex-col scroll-smooth">
      <SiteHeader />
      <div className="flex-1 max-w-5xl mx-auto w-full px-6 md:px-10">
        <div className="md:grid md:grid-cols-[15rem_1fr] md:gap-12 lg:gap-20">
          <IdentityRail user={user} />
          <div className="stagger-children min-w-0 pb-8 md:pb-20">
            <SelfieLibrarySection />
            <RaceLogSection />
          </div>
        </div>
      </div>
      <Footer />
    </main>
  );
}

function IdentityRail({ user }: { user: User }) {
  const router = useRouter();
  const { logout } = useAuth();
  const activeId = useActiveSection(JUMP_SECTIONS.map((s) => s.id));
  const memberSince = formatMemberSince(user.createdAt);
  const moreLinks = moreLinksForRole(user.role, ROUTES.PROFILE);

  function handleSignOut() {
    logout();
    router.replace(ROUTES.HOME);
  }

  return (
    <aside className="md:sticky md:top-24 md:self-start pt-10 md:pt-20 pb-8 md:pb-12 border-b md:border-0 border-line">
      <div className="flex items-start gap-5 md:block">
        <AvatarDisc name={user.name} size="md" />
        <div className="flex-1 min-w-0 md:mt-7">
          <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate">
            {user.role === "RUNNER"
              ? "Runner"
              : user.role === "PHOTOGRAPHER"
                ? "Photographer"
                : "Admin"}
            {" · Cebu · "}
            <span className="tnum">Since {memberSince}</span>
          </p>
          <p className="font-display text-2xl md:text-3xl font-medium tracking-tight leading-[1.05] text-ink mt-3">
            {user.name}
          </p>
          <p className="font-sans text-sm text-slate mt-2 break-all">
            {user.email}
          </p>
        </div>
      </div>

      <nav aria-label="Jump to section" className="hidden md:block mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          Jump to
        </p>
        <ul className="mt-4 space-y-3">
          {JUMP_SECTIONS.map((section) => {
            const isActive = activeId === section.id;
            return (
              <li key={section.id}>
                <a
                  href={`#${section.id}`}
                  aria-current={isActive ? "true" : undefined}
                  className={cn(
                    "group flex items-center gap-3 font-display text-base transition-colors",
                    isActive ? "text-ink" : "text-slate hover:text-ink",
                  )}
                >
                  <span
                    aria-hidden="true"
                    className={cn(
                      "size-1.5 rounded-full transition-colors",
                      isActive ? "bg-fresh" : "bg-line group-hover:bg-slate",
                    )}
                  />
                  <span>{section.label}</span>
                </a>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="mt-8 md:mt-12">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
          More
        </p>
        <ul className="mt-4 space-y-3">
          {moreLinks.map((link) => (
            <li key={link.href}>
              <Link
                href={link.href}
                className="font-display text-base text-slate hover:text-ink transition-colors"
              >
                {link.label}
              </Link>
            </li>
          ))}
          <li>
            <button
              type="button"
              onClick={handleSignOut}
              className="font-display text-base text-slate hover:text-ink transition-colors"
            >
              Sign out
            </button>
          </li>
        </ul>
      </div>
    </aside>
  );
}

function SelfieLibrarySection() {
  return (
    <Slab
      id="selfies"
      number="01"
      title="Selfie library"
      caption="Used by face search across every event"
    >
      <SelfieLibrary />
    </Slab>
  );
}

function RaceLogSection() {
  const savedIds = useSavedEventsStore((s) => s.ids);
  const orders = useOrdersStore((s) => s.orders);

  const log = useMemo(
    () => buildRaceLog(savedIds, orders),
    [savedIds, orders],
  );

  const trailing = `${log.length} race${log.length === 1 ? "" : "s"}`;
  return (
    <Slab id="race-log" number="02" title="Race log" trailing={trailing}>
      {log.length === 0 ? (
        <RaceLogEmpty />
      ) : (
        <ul className="border-y border-line divide-y divide-line">
          {log.map((entry) => (
            <li key={entry.id}>
              <RaceLogRow entry={entry} />
            </li>
          ))}
        </ul>
      )}
    </Slab>
  );
}

function RaceLogRow({ entry }: { entry: RaceLogEntry }) {
  const unsave = useSavedEventsStore((s) => s.unsave);
  const save = useSavedEventsStore((s) => s.save);
  const { showToast } = useToast();
  const date = formatRaceDate(entry.date);
  const isUpcoming = entry.state === "upcoming";
  const showUnsave = isUpcoming && entry.isSaved;
  const stateLabel: Record<RaceState, string> = {
    upcoming: "Saved",
    live: "Photos uploading",
    open: "Photos ready",
    past: "Archived",
  };

  const body = (
    <div className="flex items-baseline justify-between gap-6 py-6 md:py-7">
      <div className="flex-1 min-w-0">
        <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate tnum">
          {date}
        </p>
        <h3 className="font-display text-xl md:text-2xl font-medium tracking-tight text-ink mt-2 truncate">
          {entry.name}
        </h3>
        <p className="font-sans text-sm text-slate mt-2">
          {stateLabel[entry.state]}
          {!isUpcoming && entry.photosFound > 0 && (
            <>
              <span className="text-slate-soft"> · </span>
              <span className="font-mono tnum text-ink-soft">
                {entry.photosFound}
              </span>{" "}
              photos
            </>
          )}
          {entry.photosBought > 0 && (
            <>
              <span className="text-slate-soft"> · </span>
              <span className="font-mono tnum text-fresh">
                {entry.photosBought}
              </span>{" "}
              <span className="text-fresh">kept</span>
            </>
          )}
        </p>
      </div>
      {showUnsave ? (
        <button
          type="button"
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            // TODO(backend): swap for `api.delete("/me/saved-events/${entry.id}")`.
            unsave(entry.id);
            showToast({
              kind: "success",
              message: `Removed ${entry.name} from saved.`,
              action: {
                label: "Undo",
                onClick: () => save(entry.id),
              },
            });
          }}
          aria-label={`Unsave ${entry.name}`}
          className="font-sans text-sm text-slate underline decoration-line underline-offset-4 decoration-1 hover:decoration-error hover:text-error transition-colors shrink-0"
        >
          Unsave
        </button>
      ) : !isUpcoming ? (
        <span className="font-sans text-sm text-ink group-hover:text-fresh transition-colors shrink-0 inline-flex items-center gap-1.5">
          Open
          <span
            aria-hidden="true"
            className="transition-transform group-hover:translate-x-0.5"
          >
            →
          </span>
        </span>
      ) : null}
    </div>
  );

  if (isUpcoming) {
    return <div className="group block">{body}</div>;
  }

  return (
    <Link
      href={`/events/${entry.slug}`}
      aria-label={`Open ${entry.name}`}
      className="group block rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-fresh focus-visible:ring-offset-2 focus-visible:ring-offset-bone"
    >
      {body}
    </Link>
  );
}

function RaceLogEmpty() {
  return (
    <div className="border border-dashed border-line rounded-2xl p-8 md:p-12 text-center">
      <p className="font-display text-2xl md:text-3xl font-medium tracking-tight text-ink">
        No races yet.
      </p>
      <p className="font-sans text-base text-ink-soft mt-3 max-w-sm mx-auto">
        Save a race or buy a photo to start your log.
      </p>
      <Link
        href={ROUTES.EVENTS}
        className="mt-6 inline-block font-sans text-sm text-ink underline decoration-line underline-offset-4 decoration-1 hover:decoration-fresh hover:text-fresh transition-colors"
      >
        Browse races
      </Link>
    </div>
  );
}

// Race Log = saved ∪ purchased. Saved events appear immediately. Purchases
// retroactively create rows when the user bought from an event they didn't save.
// Both triggers can apply to the same event — single row, dedupe by event id.
function buildRaceLog(
  savedIds: ReadonlyArray<string>,
  orders: ReadonlyArray<{ eventId: string; photoIds: string[] }>,
): ReadonlyArray<RaceLogEntry> {
  const photosByEvent = new Map<string, number>();
  for (const order of orders) {
    photosByEvent.set(
      order.eventId,
      (photosByEvent.get(order.eventId) ?? 0) + order.photoIds.length,
    );
  }

  const includedIds = new Set<string>([
    ...savedIds,
    ...photosByEvent.keys(),
  ]);

  const entries: RaceLogEntry[] = [];
  for (const id of includedIds) {
    const event = EVENT_CATALOG.find((e) => e.id === id);
    if (!event) continue;
    entries.push({
      id: event.id,
      slug: event.slug,
      name: event.name,
      date: event.date,
      state: event.state,
      photosFound: MOCK_USER_PHOTOS_FOUND[event.id] ?? 0,
      photosBought: photosByEvent.get(event.id) ?? 0,
      isSaved: savedIds.includes(event.id),
    });
  }

  return entries.sort((a, b) => b.date.localeCompare(a.date));
}

function Slab({
  id,
  number,
  title,
  caption,
  trailing,
  children,
}: {
  id?: string;
  number: string;
  title: string;
  caption?: string;
  trailing?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      id={id}
      className="border-t border-line py-12 md:py-16 scroll-mt-24 first:border-0 first:pt-10 md:first:pt-20"
    >
      <div className="flex items-baseline justify-between gap-6 mb-8 md:mb-10">
        <div className="flex items-baseline gap-4 min-w-0">
          <span className="font-mono text-[10px] tracking-[0.15em] text-slate-soft tnum">
            {number}
          </span>
          <p className="font-mono uppercase tracking-[0.3em] text-[11px] text-ink shrink-0">
            {title}
          </p>
          {caption && (
            <p className="hidden md:block font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft truncate">
              {caption}
            </p>
          )}
        </div>
        {trailing && (
          <p className="font-mono uppercase tracking-[0.25em] text-[10px] text-slate-soft tnum shrink-0">
            {trailing}
          </p>
        )}
      </div>
      {children}
    </section>
  );
}

function Footer() {
  return (
    <footer className="px-6 md:px-10 py-8 mt-12 flex flex-col md:flex-row items-center justify-between gap-4 border-t border-line bg-bone">
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        QuickPitik &middot; Cebu, Philippines
      </p>
      <p className="font-mono uppercase tracking-[0.3em] text-[10px] text-slate-soft">
        © 2026
      </p>
    </footer>
  );
}

function useActiveSection(ids: ReadonlyArray<string>): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);
  useEffect(() => {
    const elements = ids
      .map((id) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null);
    if (elements.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
        if (visible.length > 0) {
          setActive(visible[0].target.id);
        }
      },
      {
        rootMargin: "-25% 0px -55% 0px",
        threshold: [0, 0.1, 0.5, 1],
      },
    );

    elements.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, [ids]);
  return active;
}

function moreLinksForRole(
  role: Role,
  current: string,
): ReadonlyArray<{ label: string; href: string }> {
  const all: Array<{ label: string; href: string }> = [
    { label: "Profile", href: ROUTES.PROFILE },
    { label: "Account", href: ROUTES.ACCOUNT },
  ];
  if (role === "RUNNER") all.push({ label: "Orders", href: ROUTES.ORDERS });
  if (role === "PHOTOGRAPHER")
    all.push({ label: "Dashboard", href: ROUTES.DASHBOARD });
  if (role === "ADMIN") all.push({ label: "Admin", href: ROUTES.ADMIN });
  return all.filter((l) => l.href !== current);
}

function formatMemberSince(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const year = d.getFullYear();
  return `${month} ${year}`;
}

function formatRaceDate(iso: string): string {
  const d = new Date(iso + "T00:00:00");
  if (Number.isNaN(d.getTime())) return "—";
  const month = d.toLocaleString("en-US", { month: "short" }).toUpperCase();
  const day = d.getDate().toString().padStart(2, "0");
  const year = d.getFullYear();
  return `${month} ${day} · ${year}`;
}
