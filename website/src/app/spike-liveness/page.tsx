"use client";

// AA-0.3a throwaway spike: never merged. Session id and credentials are pasted
// at runtime (from the owner's CLI) and never stored or committed.
import dynamic from "next/dynamic";
import { useState } from "react";
import { Modal } from "@/components/ui/modal";
import type { SpikeCredentials } from "./detector";

const Detector = dynamic(() => import("./detector"), { ssr: false });

export default function SpikeLivenessPage() {
  const [sessionId, setSessionId] = useState("");
  const [region, setRegion] = useState("ap-southeast-5");
  const [creds, setCreds] = useState("");
  const [open, setOpen] = useState(false);
  const [selfHost, setSelfHost] = useState(true);

  let parsed: SpikeCredentials | null = null;
  try {
    // Accepts the `Credentials` object of `aws sts assume-role` output as is.
    const c = JSON.parse(creds);
    const src = c.Credentials ?? c;
    parsed = {
      accessKeyId: src.AccessKeyId ?? src.accessKeyId,
      secretAccessKey: src.SecretAccessKey ?? src.secretAccessKey,
      sessionToken: src.SessionToken ?? src.sessionToken,
      expiration: src.Expiration ?? src.expiration,
    };
  } catch {
    parsed = null;
  }

  return (
    <main className="mx-auto max-w-xl space-y-3 p-4">
      <h1>AA-0.3a liveness spike</h1>
      <label className="block">
        Session id
        <input className="block w-full border p-2" value={sessionId} onChange={(e) => setSessionId(e.target.value)} />
      </label>
      <label className="block">
        Region
        <input className="block w-full border p-2" value={region} onChange={(e) => setRegion(e.target.value)} />
      </label>
      <label className="block">
        Credentials JSON (assume-role output)
        <textarea className="block h-32 w-full border p-2 font-mono text-xs" value={creds} onChange={(e) => setCreds(e.target.value)} />
      </label>
      <label className="block">
        <input type="checkbox" checked={selfHost} onChange={(e) => setSelfHost(e.target.checked)} /> Self-host WASM and model (off = AWS CDN)
      </label>
      <button
        type="button"
        className="border px-4 py-2"
        onClick={() => setOpen(true)}
      >
        Mount detector{parsed ? "" : " (no valid credentials: mount-only check)"}
      </button>
      <Modal isOpen={open} onClose={() => setOpen(false)} title="Live check">
        {open && (
          <Detector
            sessionId={sessionId || "spike-no-session"}
            region={region}
            selfHost={selfHost}
            credentials={
              parsed ?? { accessKeyId: "x", secretAccessKey: "x", sessionToken: "x", expiration: new Date().toISOString() }
            }
          />
        )}
      </Modal>
    </main>
  );
}
