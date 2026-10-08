const storageKey = "quickpitik-admin-action-intents";
type Intent = { actor: string; operation: string; target: string; key: string; fingerprint: string };
let records: Intent[] | undefined;
let generation = 0;

function read(): Intent[] {
  if (!records) {
    try {
      const value: unknown = JSON.parse(sessionStorage.getItem(storageKey) ?? "[]");
      records = Array.isArray(value) ? value.filter((row): row is Intent =>
        row && ["actor", "operation", "target", "key", "fingerprint"].every((field) => typeof row[field] === "string")) : [];
    } catch { records = []; }
  }
  return records;
}
function save() {
  try { sessionStorage.setItem(storageKey, JSON.stringify(read())); return true; }
  catch { return false; }
}
export function adminIntentRecoveryNotice(actor: string, operation: string, target: string) {
  return read().some((row) => row.actor === actor && row.operation === operation && row.target === target)
    ? "An earlier request is unresolved. Keep or reconstruct the original unchanged draft and cover, then retry to check it. A blank form does not mean it failed."
    : null;
}
async function hash(bytes: ArrayBuffer) {
  return Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
export async function claimAdminIntent(actor: string, operation: string, target: string, draft: object) {
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
  const existing = read().find((row) => row.actor === actor && row.operation === operation && row.target === target);
  if (existing && existing.fingerprint !== fingerprint) {
    throw new Error("An earlier request is unresolved. Reconstruct the original unchanged draft, including its cover, to check it. Do not start a replacement request.");
  }
  const intent = existing ?? { actor, operation, target, fingerprint, key: crypto.randomUUID() };
  if (!existing) read().push(intent);
  return { ...intent, reloadSafe: save() };
}
export function completeAdminIntent(intent: { key: string }) {
  records = read().filter((row) => row.key !== intent.key); save();
}
export function clearAdminActionIntents() {
  generation++; records = [];
  try { sessionStorage.removeItem(storageKey); } catch { /* In-memory recovery still resets. */ }
}
