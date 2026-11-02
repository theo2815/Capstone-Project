import { Fragment } from "react";
import Link from "next/link";
import { Kicker } from "@/components/ui/kicker";
import { RefundPolicyContent } from "@/components/orders/refund-policy-content";
import { REFUND_POLICY_MARKER, type LegalBlock, type LegalDoc } from "@/lib/legal/types";
import { LEGAL_LAST_UPDATED, LEGAL_VERSION } from "@/lib/legal/version";

const TEXT_LINK =
  "text-ink underline decoration-line-strong underline-offset-4 hover:decoration-fresh transition-colors";

// Renders **bold** and [label](href). Site paths use next/link; anchors,
// mailto and outside links use a plain <a>.
export function LegalText({ text }: { text: string }) {
  return text.split(/(\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/).map((part, i) => {
    const bold = part.match(/^\*\*([^*]+)\*\*$/);
    if (bold) return <strong key={i} className="font-semibold text-ink">{bold[1]}</strong>;
    const link = part.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
    if (!link) return <Fragment key={i}>{part}</Fragment>;
    const [, label, href] = link;
    return href.startsWith("/") ? (
      <Link key={i} href={href} className={TEXT_LINK}>{label}</Link>
    ) : (
      <a key={i} href={href} className={TEXT_LINK}>{label}</a>
    );
  });
}

function Block({ block }: { block: LegalBlock }) {
  if (block === REFUND_POLICY_MARKER) return <RefundPolicyContent />;
  if (typeof block === "string") return <p><LegalText text={block} /></p>;
  if ("table" in block) {
    const [head, ...rows] = block.table;
    return (
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr>
              {head.map((cell) => (
                <th key={cell} className="border-b border-line-strong py-2 pr-4 font-semibold text-ink">{cell}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row[0]}>
                {row.map((cell, i) => (
                  <td key={i} className="border-b border-line py-3 pr-4 align-top"><LegalText text={cell} /></td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }
  const items = block.list.map((item) => <li key={item}><LegalText text={item} /></li>);
  return block.start === undefined ? (
    <ul className="list-disc space-y-2 pl-5">{items}</ul>
  ) : (
    <ol start={block.start} className="list-decimal space-y-2 pl-6">{items}</ol>
  );
}

// A quiet reading page: no client JS, anchors on every section.
export function LegalDocument({ doc }: { doc: LegalDoc }) {
  return (
    <article className="max-w-[68ch]">
      <Kicker as="p">Legal</Kicker>
      <h1 className="mt-3 font-display text-4xl font-extrabold tracking-tight text-ink md:text-5xl">{doc.title}</h1>
      <p className="mt-4 font-mono text-[14px] text-slate tnum">
        Version {LEGAL_VERSION} · Last updated {LEGAL_LAST_UPDATED}
      </p>
      {doc.sections.map((section) => (
        <section key={section.id} id={section.id} className="mt-12 scroll-mt-24">
          <h2 className="font-display text-xl font-bold tracking-tight text-ink md:text-2xl">{section.heading}</h2>
          <div className="mt-4 space-y-4 font-sans text-base leading-relaxed text-ink-soft">
            {section.blocks.map((block, i) => <Block key={i} block={block} />)}
          </div>
        </section>
      ))}
    </article>
  );
}
