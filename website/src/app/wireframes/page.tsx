import Link from "next/link";

const sections = [
  {
    module: "Module 1",
    label: "Mobile Application — Photographer + Runner",
    items: [
      { id: "UC-M1-1.1", name: "Tether Camera to Mobile App", href: "/wireframes/m1-1-tether", traces: "SO1.1" },
      { id: "UC-M1-1.2", name: "Capture & Queue Photo for Upload", href: "/wireframes/m1-2-capture", traces: "SO1.1" },
      { id: "UC-M1-1.3", name: "Auto-Upload to Cloud", href: "/wireframes/m1-3-upload", traces: "SO1.1, SO1.2" },
      { id: "UC-M1-1.4", name: "Local-Cache During Signal Loss", href: "/wireframes/m1-4-offline", traces: "SO1.2" },
      { id: "UC-M1-1.5", name: "Receive Photo-Found Notification", href: "/wireframes/m1-5-notification", traces: "SO1.3" },
    ],
  },
  {
    module: "Module 2",
    label: "Desktop Application — BatchMyPhotos",
    items: [
      { id: "UC-M2-2.1", name: "Sync Event Library", href: "/wireframes/m2-1-sync", traces: "SO2.3" },
      { id: "UC-M2-2.2", name: "Run Blur Detection", href: "/wireframes/m2-2-blur", traces: "SO2.1" },
      { id: "UC-M2-2.3", name: "Auto-Sort into Batch Folders", href: "/wireframes/m2-3-sort", traces: "SO2.2" },
      { id: "UC-M2-2.4", name: "Upload Sorted Batch to Backend", href: "/wireframes/m2-4-upload", traces: "SO2.3" },
    ],
  },
  {
    module: "Module 3",
    label: "Marketplace & AI Search — Web / Mobile",
    items: [
      { id: "UC-M3-3.1", name: "Register / Login", href: "/wireframes/m3-1-auth", traces: "supports GO3" },
      { id: "UC-M3-3.2", name: "Browse Events", href: "/wireframes/m3-2-events", traces: "supports GO3" },
      { id: "UC-M3-3.3", name: "Search by Selfie", href: "/wireframes/m3-3-selfie", traces: "SO3.1, SO3.2" },
      { id: "UC-M3-3.4", name: "Search by Bib Number", href: "/wireframes/m3-4-bib", traces: "SO3.1, SO3.2" },
      { id: "UC-M3-3.5", name: "Preview Photo (Watermarked)", href: "/wireframes/m3-5-preview", traces: "SO3.3" },
      { id: "UC-M3-3.6", name: "Add to Cart", href: "/wireframes/m3-6-cart", traces: "SO3.3" },
      { id: "UC-M3-3.7", name: "Checkout & Pay", href: "/wireframes/m3-7-checkout", traces: "SO3.3" },
      { id: "UC-M3-3.8", name: "Download Purchased Photos", href: "/wireframes/m3-8-download", traces: "SO3.3" },
    ],
  },
];

export default function WireframesIndex() {
  return (
    <div className="min-h-screen bg-[#FAFAF7] py-12 px-6">
      <div className="mx-auto max-w-5xl">
        <div className="mb-10 border-b border-neutral-900 pb-6">
          <div className="font-mono text-[10px] uppercase tracking-[0.3em] text-neutral-500">
            QuickPitik · Capstone SRS · Figures appendix
          </div>
          <h1 className="mt-2 font-display text-4xl font-semibold text-neutral-900">
            Wireframes — SRS §3.2 Modules 1, 2, 3
          </h1>
          <p className="mt-3 max-w-2xl text-sm text-neutral-600">
            Low-fidelity wireframes for the 17 transactions specified in SRS §3.2.
            Each figure satisfies the &quot;Must show&quot; contract declared in the
            corresponding wireframe placeholder. Open a figure, screenshot it,
            and paste it into the SRS section that owns the use case.
          </p>
        </div>

        <div className="space-y-10">
          {sections.map((s) => (
            <section key={s.module}>
              <div className="mb-3 flex items-baseline gap-3">
                <h2 className="font-display text-xl font-semibold text-neutral-900">
                  {s.module}
                </h2>
                <span className="font-mono text-[11px] uppercase tracking-wider text-neutral-500">
                  {s.label}
                </span>
              </div>
              <ul className="divide-y divide-neutral-200 border-y border-neutral-200">
                {s.items.map((it) => (
                  <li key={it.id}>
                    <Link
                      href={it.href}
                      className="grid grid-cols-12 items-center gap-2 px-2 py-3 hover:bg-white"
                    >
                      <span className="col-span-3 font-mono text-xs text-neutral-500">
                        {it.id}
                      </span>
                      <span className="col-span-7 text-sm text-neutral-900">
                        {it.name}
                      </span>
                      <span className="col-span-2 text-right font-mono text-[10px] uppercase tracking-wider text-neutral-500">
                        {it.traces}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>

        <div className="mt-12 rounded border border-dashed border-neutral-300 bg-white p-4">
          <div className="font-mono text-[10px] uppercase tracking-wider text-neutral-500">
            Figure-capture tip
          </div>
          <p className="mt-2 text-sm text-neutral-700">
            Each wireframe page is sized so the device frame, the &quot;must show&quot;
            checklist, and the figure caption all fit in a single
            screenshot at 1280 × 1024 zoom 100&nbsp;%. Use Windows{" "}
            <kbd className="rounded border border-neutral-300 bg-neutral-50 px-1.5 py-0.5 font-mono text-[10px]">
              Win + Shift + S
            </kbd>{" "}
            to clip just the wireframe stage.
          </p>
        </div>
      </div>
    </div>
  );
}
