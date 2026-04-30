"use client";

import { useState } from "react";
import {
  WireframeAnnotation,
  WireframeNav,
  WireframeStamp,
} from "../_components/WireframeNav";

const ACCENT = "#FF4A1C";

const runnerChapters = [
  { ch: "01", t: "00:00", title: "Pick your event", note: "Filter by date, city, distance" },
  { ch: "02", t: "00:18", title: "Selfie or bib?", note: "Both work. One is faster — your call." },
  { ch: "03", t: "00:42", title: "Watch the AI match", note: "Live thumbnails as they get found" },
  { ch: "04", t: "01:10", title: "Buy & download", note: "GCash · Maya · card" },
];

const photographerChapters = [
  { ch: "01", t: "00:00", title: "How big is your event?", note: "Volume picker → routes to the right tool" },
  { ch: "02", t: "00:24", title: "Connect or upload", note: "Mobile / Web / Desktop — same gallery" },
  { ch: "03", t: "01:00", title: "Watch the cull", note: "Live blur scoring as photos arrive" },
  { ch: "04", t: "01:38", title: "Gallery goes live", note: "Searchable to runners in minutes" },
];

export default function ChannelSwitchWireframe() {
  const [channel, setChannel] = useState<"01" | "02">("01");

  return (
    <main className="min-h-screen bg-[#0E0E0E] text-neutral-100">
      <WireframeNav active="channel-switch" />
      <WireframeStamp label="WF · 03 · Channel Switch · Broadcast" />

      <section className="border-b border-dashed border-neutral-700">
        <div className="mx-auto max-w-6xl px-6 py-6">
          <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            ▸ Section A — Channel selector chrome (sits above the hero, always
            visible)
          </p>
        </div>

        <div className="mx-auto max-w-6xl border-y border-neutral-800 bg-[#161616] px-6 py-4">
          <div className="flex flex-wrap items-center justify-between gap-4 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]">
            <div className="flex items-center gap-3">
              <span
                className="h-2 w-2 rounded-full"
                style={{ background: ACCENT, animation: "rec-pulse 1.4s ease-in-out infinite" }}
              />
              <span style={{ color: ACCENT }}>● REC</span>
              <span className="text-neutral-500">QUICK PITIK · LIVE</span>
            </div>

            <div className="flex items-center gap-1">
              <ChannelButton
                ch="01"
                label="Runner Feed"
                active={channel === "01"}
                onClick={() => setChannel("01")}
              />
              <ChannelButton
                ch="02"
                label="Photographer Feed"
                active={channel === "02"}
                onClick={() => setChannel("02")}
              />
            </div>

            <div className="flex items-center gap-3 text-neutral-500">
              <span>NOW · {channel === "01" ? "FIND YOUR PHOTOS" : "GET YOUR SHOOT ONLINE"}</span>
              <span className="tnum">04:30:22</span>
            </div>
          </div>
        </div>

        <div className="mx-auto max-w-6xl px-6 pb-3 pt-3">
          <div className="overflow-hidden border border-dashed border-neutral-700 bg-[#0E0E0E]">
            <div className="flex animate-marquee gap-8 whitespace-nowrap py-2 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-400">
              {Array.from({ length: 6 }).map((_, i) => (
                <span key={i}>
                  ◆ Cebu Marathon · 12,432 photos · 8 photographers · live now
                  &nbsp;&nbsp;◆ Avg search 2.8s &nbsp;&nbsp;◆ Blur cull 99.4%
                  precision &nbsp;&nbsp;
                </span>
              ))}
            </div>
          </div>
          <p className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            [Ticker — same on both channels. Live event roll, recent stats.]
          </p>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-700">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section B — Hero changes per channel. Big Shoulders for runner
            (broadcast/kinetic), Fraunces italic supports for photographer
            (editorial/craft).
          </WireframeAnnotation>

          <div className="mt-8 border border-dashed border-neutral-700 bg-[#161616] p-8">
            {channel === "01" ? <RunnerHero /> : <PhotographerHero />}
          </div>

          <p className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            [Interaction] Click CH 01 / CH 02 above. Page swaps with brief
            CRT-style transition (scanline wipe + chromatic shift, ~280ms).
          </p>
        </div>
      </section>

      <section className="border-b border-dashed border-neutral-700">
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section C — Chapter list (the &ldquo;explainer reel&rdquo;)
          </WireframeAnnotation>

          <div className="mt-8 grid gap-6 md:grid-cols-[1fr_2fr]">
            <div className="border border-dashed border-neutral-700 p-5">
              <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                Currently watching
              </p>
              <p className="mt-2 font-[family-name:var(--font-display)] text-2xl font-black uppercase tracking-tight">
                CH {channel} · {channel === "01" ? "Runner Feed" : "Photographer Feed"}
              </p>
              <div className="mt-4 aspect-video border border-dashed border-neutral-700 bg-[#0E0E0E]">
                <div className="flex h-full items-center justify-center text-neutral-700">
                  <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]">
                    [autoplaying explainer · scrubbable]
                  </span>
                </div>
              </div>
              <div className="mt-3 flex justify-between font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-500">
                <span>00:42 / 01:38</span>
                <span style={{ color: ACCENT }}>● ON AIR</span>
              </div>
            </div>

            <ol className="space-y-2">
              {(channel === "01" ? runnerChapters : photographerChapters).map(
                (c, i) => (
                  <li
                    key={c.ch}
                    className={`flex items-baseline gap-4 border border-dashed border-neutral-700 p-4 transition-colors hover:border-[${ACCENT}] ${
                      i === 0 ? "bg-[#161616]" : ""
                    }`}
                  >
                    <span
                      className="font-[family-name:var(--font-display)] text-3xl font-black tracking-tight"
                      style={{ color: i === 0 ? ACCENT : "#666" }}
                    >
                      {c.ch}
                    </span>
                    <div className="flex-1">
                      <div className="flex items-baseline justify-between gap-2">
                        <p className="font-[family-name:var(--font-display)] text-lg font-bold uppercase tracking-tight">
                          {c.title}
                        </p>
                        <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                          {c.t}
                        </span>
                      </div>
                      <p className="mt-1 font-[family-name:var(--font-editorial)] text-sm italic text-neutral-400">
                        {c.note}
                      </p>
                    </div>
                  </li>
                )
              )}
            </ol>
          </div>
        </div>
      </section>

      {channel === "02" && (
        <section className="border-b border-dashed border-neutral-700">
          <div className="mx-auto max-w-6xl px-6 py-12">
            <WireframeAnnotation>
              ▸ Section D — Embedded inside Chapter 01 (CH 02 only): Volume
              question. Same fork pattern as #1 and #2 — shared across all
              wireframes.
            </WireframeAnnotation>

            <div className="mt-6 border border-dashed border-neutral-700 bg-[#161616] p-6">
              <p className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
                Chapter 01 · Inline chooser
              </p>
              <p className="mt-2 font-[family-name:var(--font-display)] text-2xl font-bold uppercase tracking-tight">
                How big is the event you&apos;re shooting?
              </p>

              <div className="mt-4 grid gap-3 md:grid-cols-3">
                {[
                  { label: "< 500", app: "Web upload", time: "Browser, no install" },
                  { label: "500–10k", app: "Mobile app", time: "Camera tether, auto-upload" },
                  { label: "10k+", app: "Desktop", time: "Auto-batches into 500s" },
                ].map((opt) => (
                  <div
                    key={opt.label}
                    className="border border-dashed border-neutral-600 p-4 transition-colors hover:border-[#FF4A1C]"
                  >
                    <p className="font-[family-name:var(--font-mono)] text-[10px] uppercase tracking-[0.22em] text-neutral-500">
                      Photos
                    </p>
                    <p className="mt-1 font-[family-name:var(--font-display)] text-3xl font-black tracking-tight">
                      {opt.label}
                    </p>
                    <p
                      className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.18em]"
                      style={{ color: ACCENT }}
                    >
                      → {opt.app}
                    </p>
                    <p className="mt-1 font-[family-name:var(--font-editorial)] text-xs italic text-neutral-400">
                      {opt.time}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}

      <section>
        <div className="mx-auto max-w-6xl px-6 py-12">
          <WireframeAnnotation>
            ▸ Section E — Persistent channel indicator (post-hero, fixed top-right)
          </WireframeAnnotation>
          <div className="mt-3 inline-flex items-center gap-2 border border-dashed border-neutral-700 bg-[#161616] px-3 py-2">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: ACCENT, animation: "rec-pulse 1.4s ease-in-out infinite" }}
            />
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]">
              Watching · CH {channel}
            </span>
            <span className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
              · flip channel
            </span>
          </div>
          <p className="mt-3 font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-500">
            [Why] One-click switch removes the dread of &ldquo;did I pick the
            wrong path?&rdquo; Channels are a familiar mental model — TV.
          </p>
        </div>
      </section>
    </main>
  );
}

function ChannelButton({
  ch,
  label,
  active,
  onClick,
}: {
  ch: string;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-2 border px-3 py-1.5 transition-colors ${
        active
          ? "border-[#FF4A1C] bg-[#FF4A1C] text-black"
          : "border-dashed border-neutral-700 text-neutral-400 hover:border-neutral-400"
      }`}
    >
      <span className="tnum font-bold">CH {ch}</span>
      <span>{label}</span>
    </button>
  );
}

function RunnerHero() {
  return (
    <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
      <div>
        <p
          className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
          style={{ color: ACCENT }}
        >
          CH 01 · LIVE
        </p>
        <h1 className="mt-3 font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight md:text-6xl">
          Find your finish.
          <br />
          <span className="text-neutral-500">In under three seconds.</span>
        </h1>
        <p className="mt-4 max-w-md font-[family-name:var(--font-editorial)] text-lg italic text-neutral-300">
          Selfie or bib. Either way — your photos, in seconds.
        </p>
        <button
          className="mt-6 border-2 px-4 py-2 font-[family-name:var(--font-mono)] text-[12px] uppercase tracking-[0.22em]"
          style={{ borderColor: ACCENT, color: ACCENT }}
        >
          Find my photos · 30s
        </button>
      </div>
      <div className="flex aspect-square items-center justify-center border border-dashed border-neutral-700 bg-[#0E0E0E] font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-700">
        [hero photo · runner mid-stride]
      </div>
    </div>
  );
}

function PhotographerHero() {
  return (
    <div className="grid gap-6 md:grid-cols-[2fr_1fr]">
      <div>
        <p
          className="font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em]"
          style={{ color: ACCENT }}
        >
          CH 02 · LIVE
        </p>
        <h1 className="mt-3 font-[family-name:var(--font-display)] text-5xl font-black uppercase leading-[0.92] tracking-tight md:text-6xl">
          Sell every keeper.
          <br />
          <span className="font-[family-name:var(--font-editorial)] italic text-neutral-300">
            Cull every blur.
          </span>
        </h1>
        <p className="mt-4 max-w-md font-[family-name:var(--font-editorial)] text-lg italic text-neutral-300">
          From shutter to gallery in minutes — at any volume.
        </p>
        <button
          className="mt-6 border-2 px-4 py-2 font-[family-name:var(--font-mono)] text-[12px] uppercase tracking-[0.22em]"
          style={{ borderColor: ACCENT, color: ACCENT }}
        >
          Get my shoot online · 2 min
        </button>
      </div>
      <div className="flex aspect-square items-center justify-center border border-dashed border-neutral-700 bg-[#0E0E0E] font-[family-name:var(--font-mono)] text-[11px] uppercase tracking-[0.22em] text-neutral-700">
        [hero photo · long lens]
      </div>
    </div>
  );
}
