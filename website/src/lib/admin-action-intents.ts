const storageKey = "quickpitik-admin-action-intents";
type Intent = { actor: string; operation: string; target: string; key: string; fingerprint: string; sentAt: number };
// The server shows a request only once it is saved, so "no record" is believed only after the request can no
// longer be running: past both browser timeouts (30 s, 120 s with a file; lib/api.ts) and the server's hold on
// a claimed request (AdminActionProperties lease, 5 minutes by default).
const NO_RECORD_WAIT_MS = 5 * 60_000;
let records: Intent[] | undefined;
let generation = 0;

function read(): Intent[] {
  if (!records) {
    try {
      const value: unknown = JSON.parse(sessionStorage.getItem(storageKey) ?? "[]");
      // A record written before send times existed counts as sent just now.
      records = Array.isArray(value) ? value.filter((row): row is Intent =>
        row && ["actor", "operation", "target", "key", "fingerprint"].every((field) => typeof row[field] === "string"))
        .map((row) => ({ ...row, sentAt: typeof row.sentAt === "number" ? row.sentAt : Date.now() })) : [];
    } catch { records = []; }
  }
  return records;
}
function save() {
  try { sessionStorage.setItem(storageKey, JSON.stringify(read())); return true; }
  catch { return false; }
}
export function adminIntentRecoveryNotice(actor: string, operation: string, target: string) {
  const intent = read().find((row) => row.actor === actor && row.operation === operation && row.target === target);
  if (!intent) return null;
  const guidance = operation === "event.update"
    ? "Retry only with the original edit snapshot still open. After reload or a changed event, the original revision and submitted fields cannot be reconstructed here. Saving again checks the earlier request first; if it is still open, ask the project maintainer to reconcile it."
    : `Keep or reconstruct the original unchanged draft${operation === "event.create" ? " and cover" : ""}, then retry to check it. A blank form does not mean it failed.`;
  return `An earlier request is unresolved. ${guidance} If the server has no record of it, a different request is held back for about ${NO_RECORD_WAIT_MS / 60_000} minutes after it was sent. Request reference: ${intent.key}.`;
}
async function hash(bytes: ArrayBuffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
/** What the server says happened to an earlier key: it ran, it was refused for good, it has no record of it, or it cannot say yet. */
export type AdminIntentOutcome = "succeeded" | "failed" | "missing" | "open";
export async function claimAdminIntent(actor: string, operation: string, target: string, draft: object,
  resolve?: (key: string) => Promise<AdminIntentOutcome>) {
  const started = generation;
  const fields = await Promise.all(Object.entries(draft).sort(([a], [b]) => a.localeCompare(b)).filter(([, value]) => value !== undefined).map(async ([name, value]) => {
    if (value instanceof Blob) {
      const bytes = await new Promise<ArrayBuffer>((resolve, reject) => {
        const reader = new FileReader(); reader.onload = () => resolve(reader.result as ArrayBuffer); reader.onerror = () => reject(reader.error); reader.readAsArrayBuffer(value);
      });
      return [name, await hash(bytes)];
    }
    return [name, typeof value === "string" ? value.trim() : value];
  }));
  const fingerprint = await hash(new TextEncoder().encode(JSON.stringify(fields)).buffer);
  if (started !== generation) throw new Error("Session changed. Reopen the form.");
  let existing = read().find((row) => row.actor === actor && row.operation === operation && row.target === target);
  if (existing && existing.fingerprint !== fingerprint) {
    // Fail closed: only a definite answer from the server releases the earlier key.
    const outcome = resolve ? await resolve(existing.key).catch(() => "open" as const) : "open";
    if (started !== generation) throw new Error("Session changed. Reopen the form.");
    const waitMs = existing.sentAt + NO_RECORD_WAIT_MS - Date.now();
    if (outcome === "missing" && waitMs > 0) {
      const minutes = Math.ceil(waitMs / 60_000);
      throw new Error(`An earlier request may still be on its way: the server has no record of it yet, so this one was not sent. Try again in about ${minutes} minute${minutes === 1 ? "" : "s"}. Request reference: ${existing.key}.`);
    }
    if (outcome === "open") {
      throw new Error(`An earlier request is still unresolved, so this one was not sent. Reconstruct the original unchanged draft to retry it, or ask the project maintainer to reconcile it (admin manual, "Unresolved admin request"). Request reference: ${existing.key}.`);
    }
    completeAdminIntent(existing);
    existing = undefined;
  }
  const intent = existing ?? { actor, operation, target, fingerprint, key: crypto.randomUUID(), sentAt: 0 };
  intent.sentAt = Date.now(); // A retry of the same draft sends it again.
  if (!existing) read().push(intent);
  return { ...intent, reused: !!existing, reloadSafe: save() };
}
export function completeAdminIntent(intent: { key: string }) {
  records = read().filter((row) => row.key !== intent.key); save();
}
export function clearAdminActionIntents() {
  generation++; records = [];
  try { sessionStorage.removeItem(storageKey); } catch { /* In-memory recovery still resets. */ }
}
