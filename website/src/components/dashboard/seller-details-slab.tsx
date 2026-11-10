"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import LiveSelfieCheck, { type LiveSelfieCheckProps, type LiveSelfieErrorKind } from "@/components/dashboard/live-selfie-check";
import { LegalText } from "@/components/legal/legal-document";
import { Slab } from "@/components/profile-shell/slab";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { Skeleton } from "@/components/ui/skeleton";
import { useSellerDetails } from "@/hooks/use-seller-details";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import {
  isValidIdNumber,
  isValidPhone,
  reviewStateOf,
  SELLER_FILE_MAX_BYTES,
  SELLER_FILE_TYPES,
  SELLER_ID_TYPES,
  startLivenessSession,
  submitLiveCheck,
  submitSellerDetails,
  type SellerDetails,
  type SellerIdType,
} from "@/lib/api-seller-details";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { LEGAL_COPY } from "@/lib/legal/copy";
import { cn } from "@/lib/utils";

type FileField = "idFront" | "idBack" | "selfie" | "birForm";
type Errors = Partial<Record<string, string>>;
const FIELDS = ["legalName", "idType", "idNumber", "idFront", "idBack", "selfie", "address", "phone", "birForm", "biometricConsent"];

const INPUT_CLS =
  "w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors";
const LABEL_CLS = "font-sans text-sm text-slate";
const idLabel = (value?: string | null) => SELLER_ID_TYPES.find((t) => t.value === value)?.label ?? "";

// Settings section 09 (SC-2a): the photographer's seller details, which the
// backend needs before new uploads (SELLER_DETAILS_REQUIRED).
export function SellerDetailsSlab() {
  const queryClient = useQueryClient();
  const { query, queryKey } = useSellerDetails();
  const [resending, setResending] = useState(false);
  const details = query.data;
  const state = details ? reviewStateOf(details) : null;
  const onFile = state === "manual_review" || state === "auto_approved" || state === "approved";
  const saved = (next: SellerDetails) => queryClient.setQueryData(queryKey, next);

  return (
    <Slab id="seller-details" number="09" title="Seller details" caption="Required before you upload">
      <p className="max-w-xl font-sans text-sm text-slate leading-relaxed">
        The Internet Transactions Act and BIR rules ask online marketplaces to collect these. Only QuickPitik
        admins see them; we never show your ID, these photos or your phone number publicly.{" "}
        <Link href={ROUTES.PRIVACY} target="_blank" rel="noopener noreferrer" className="text-ink underline decoration-line-strong underline-offset-4 hover:decoration-fresh">
          Privacy Policy
        </Link>
      </p>
      <div className="mt-8 max-w-xl">
        {query.isPending ? (
          <Skeleton className="h-24 w-full rounded-2xl" />
        ) : query.isError ? (
          <div role="alert" className="font-sans text-sm text-ink">
            <p>We couldn&apos;t load your seller details.</p>
            <button type="button" onClick={() => query.refetch()} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-4")}>
              Try again
            </button>
          </div>
        ) : details && onFile && !resending ? (
          <div>
            <p className="font-sans text-base font-semibold text-ink">
              {state === "manual_review" ? "We're checking your details. You can upload meanwhile." : "Approved"}
            </p>
            <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 font-sans text-sm">
              <dt className="text-slate">Legal name</dt><dd className="text-ink">{details.legalName}</dd>
              <dt className="text-slate">ID</dt>
              <dd className="text-ink">{idLabel(details.idType)}</dd>
              <dt className="text-slate">Address</dt><dd className="text-ink break-words">{details.address}</dd>
              <dt className="text-slate">Phone</dt><dd className="text-ink tnum">{details.phone}</dd>
              <dt className="text-slate">BIR Form 2303</dt><dd className="text-ink">{details.hasBirForm ? "Sent" : "Not sent"}</dd>
            </dl>
            {state === "manual_review" && <LiveCheckPanel onSent={saved} />}
            <button type="button" onClick={() => setResending(true)} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-6")}>
              Send new details
            </button>
          </div>
        ) : (
          <>
            {state === "rejected" && (
              <div role="status" className="mb-8 rounded-xl border border-error/40 bg-error/5 px-5 py-4 font-sans text-sm text-ink">
                <p className="font-semibold">Your seller details were not accepted.</p>
                <p className="mt-1">Reason: {details?.rejectionReason}</p>
                <p className="mt-1">New uploads are paused until you send them again.</p>
              </div>
            )}
            <SellerDetailsForm
              initial={details}
              onCancel={resending ? () => setResending(false) : undefined}
              onSent={(next) => {
                saved(next);
                setResending(false);
              }}
            />
          </>
        )}
      </div>
    </Slab>
  );
}

// ─── The live selfie (AA-I14) ───────────────────────────────────────────

class LiveCheckStopped extends Error {
  constructor(readonly kind: LiveSelfieErrorKind | "cancel") {
    super(kind);
  }
}

const LIVE_COPY: Record<string, string> = {
  LIVENESS_ATTEMPTS_EXHAUSTED: "You've used today's 3 tries. Try again tomorrow, or have an admin check instead.",
  LIVENESS_UNAVAILABLE: "The live check isn't available right now. Have an admin check instead.",
  LIVENESS_SESSION_INVALID: "That live check ran out. Take it again.",
  camera: "We couldn't use your camera. Allow camera access and take it again, or have an admin check instead.",
  connection: "The live check lost its connection or ran out. Take it again.",
  timeout: "The live check didn't finish in time. Take it again, or have an admin check instead.",
  other: "The live check didn't finish. Take it again, or have an admin check instead.",
  load: "The live check isn't available right now. Have an admin check instead.",
};

// Starts the session only on press (it expires about 3 minutes after creation,
// AA-0.3b), shows the check, and resolves with the session id once it completes.
// The browser never sees a score; the server reads the result at submit.
function useLiveSelfie() {
  const [check, setCheck] = useState<Omit<LiveSelfieCheckProps, "onError" | "onCancel"> & { stop: (kind: LiveCheckStopped["kind"]) => void } | null>(null);
  const run = async (): Promise<string> => {
    const session = await startLivenessSession();
    const { sessionId, region, credentials } = session;
    if (!credentials) throw new LiveCheckStopped("load"); // only under Cognito, which AA-0.2 did not choose
    return new Promise((resolve, reject) =>
      setCheck({
        sessionId, region, credentials,
        onComplete: () => { setCheck(null); resolve(sessionId); },
        stop: (kind) => { setCheck(null); reject(new LiveCheckStopped(kind)); },
      }),
    );
  };
  const modal = check && <LiveSelfieCheck {...check} onError={check.stop} onCancel={() => check.stop("cancel")} />;
  return { run, modal };
}

function errorsFrom(err: unknown): Errors {
  if (err instanceof LiveCheckStopped) return err.kind === "cancel" ? {} : { form: LIVE_COPY[err.kind] };
  if (!(err instanceof ApiError)) return { form: "We couldn't send your details. Check your connection and try again." };
  const server: Errors = {};
  for (const e of err.errors) {
    if (LIVE_COPY[e.code]) server.form = LIVE_COPY[e.code];
    else server[e.field && FIELDS.includes(e.field) ? e.field : "form"] = e.message;
  }
  return server;
}

const idNumberError = (value: string) =>
  isValidIdNumber(value) ? undefined : "Enter the ID number as printed on your ID (4 to 30 letters or numbers).";
const CONSENT_ERROR = "Tick the box to agree before the live selfie.";
const only = (found: Errors): Errors => Object.fromEntries(Object.entries(found).filter(([, v]) => v));

const fieldProps = (field: string, errors: Errors) => ({
  id: `seller-${field}`,
  "aria-invalid": errors[field] ? true : undefined,
  "aria-describedby": errors[field] ? `seller-${field}-error` : undefined,
});

function IdNumberField({ value, onChange, errors, disabled }: { value: string; onChange: (v: string) => void; errors: Errors; disabled: boolean }) {
  return (
    <div className="flex flex-col gap-2">
      <label htmlFor="seller-idNumber" className={LABEL_CLS}>ID number (as printed on your ID)</label>
      <p className="font-sans text-sm text-slate-soft">Spaces and dashes are fine. We never show it back.</p>
      <input {...fieldProps("idNumber", errors)} value={value} maxLength={40} autoComplete="off" spellCheck={false} disabled={disabled}
        onChange={(e) => onChange(e.target.value)} className={cn(INPUT_CLS, "tnum")} />
      <FieldError message={errors.idNumber} id="seller-idNumber-error" density="tight" />
    </div>
  );
}

function ConsentTick({ checked, onChange, errors, disabled }: { checked: boolean; onChange: (v: boolean) => void; errors: Errors; disabled: boolean }) {
  return (
    <div className="rounded-xl border border-line bg-bone-deep px-5 py-4 font-sans text-sm leading-relaxed text-ink-soft">
      <p><LegalText text={LEGAL_COPY.identityNotice} newTab /></p>
      <label className="mt-3 flex min-h-11 cursor-pointer items-start gap-3 text-ink select-none">
        <input {...fieldProps("biometricConsent", errors)} type="checkbox" checked={checked} disabled={disabled}
          onChange={(e) => onChange(e.target.checked)} className="mt-1 size-4 shrink-0 accent-ink" />
        <span>{LEGAL_COPY.identityTickLabel}</span>
      </label>
      <FieldError message={errors.biometricConsent} id="seller-biometricConsent-error" density="tight" />
    </div>
  );
}

// A manual-review record takes only the live check (C2b): no files again.
function LiveCheckPanel({ onSent }: { onSent: (saved: SellerDetails) => void }) {
  const { showToast } = useToast();
  const live = useLiveSelfie();
  const [idNumber, setIdNumber] = useState("");
  const [consent, setConsent] = useState(false);
  const [errors, setErrors] = useState<Errors>({});
  const [sending, setSending] = useState(false);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (sending) return;
    const found = only({ idNumber: idNumberError(idNumber), biometricConsent: consent ? undefined : CONSENT_ERROR });
    setErrors(found);
    if (Object.keys(found).length > 0) return;
    const isCurrent = captureAuthSession();
    setSending(true);
    try {
      const livenessSessionId = await live.run();
      if (!isCurrent()) return;
      const saved = await submitLiveCheck({ livenessSessionId, idNumber: idNumber.trim() });
      if (!isCurrent()) return;
      const approved = reviewStateOf(saved) === "auto_approved";
      showToast({ kind: "success", message: approved ? "Seller details approved." : "Live check sent. An admin will check your details." });
      setIdNumber(""); // never shown back after a send
      setConsent(false);
      onSent(saved);
    } catch (err) {
      if (isCurrent()) setErrors(errorsFrom(err));
    } finally {
      if (isCurrent()) setSending(false);
    }
  };

  return (
    <form onSubmit={submit} noValidate className="mt-8 flex flex-col gap-6 border-t border-line pt-6">
      <p className="font-sans text-sm text-slate">Take a live selfie to finish the check now, without waiting for an admin.</p>
      <IdNumberField value={idNumber} onChange={(v) => { setIdNumber(v); setErrors({}); }} errors={errors} disabled={sending} />
      <ConsentTick checked={consent} onChange={(v) => { setConsent(v); setErrors({}); }} errors={errors} disabled={sending} />
      <FieldError message={errors.form} id="seller-form-error" density="tight" />
      <button type="submit" disabled={sending} className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-fit")}>
        {sending ? "Checking your details…" : "Take the live selfie now"}
      </button>
      {live.modal}
    </form>
  );
}

// ─── The full form ──────────────────────────────────────────────────────

function fileError(file: File | null, required: string | null): string | null {
  if (!file) return required;
  if (!SELLER_FILE_TYPES.includes(file.type)) return "Use a JPEG, PNG or WebP photo.";
  if (file.size > SELLER_FILE_MAX_BYTES) return "Photo must be 5 MB or smaller.";
  return null;
}

function SellerDetailsForm({
  initial,
  onSent,
  onCancel,
}: {
  initial?: SellerDetails;
  onSent: (saved: SellerDetails) => void;
  onCancel?: () => void;
}) {
  const { showToast } = useToast();
  const live = useLiveSelfie();
  const [legalName, setLegalName] = useState(initial?.legalName ?? "");
  const [idType, setIdType] = useState<SellerIdType | "">(initial?.idType ?? "");
  const [idNumber, setIdNumber] = useState("");
  const [address, setAddress] = useState(initial?.address ?? "");
  const [phone, setPhone] = useState(initial?.phone ?? "");
  const [files, setFiles] = useState<Record<FileField, File | null>>({ idFront: null, idBack: null, selfie: null, birForm: null });
  const [consent, setConsent] = useState(false);
  const [errors, setErrors] = useState<Errors>({});
  const [sending, setSending] = useState<"live" | "admin" | null>(null);
  const busy = sending !== null;
  const philsys = idType === "philsys";

  const clear = (field: string) => setErrors((prev) => ({ ...prev, [field]: undefined, form: undefined }));
  const pick = (field: FileField, file: File | null) => {
    setFiles((prev) => ({ ...prev, [field]: file }));
    clear(field);
  };

  const validate = (withLive: boolean): Errors => {
    const name = legalName.trim();
    const addr = address.trim();
    return only({
      legalName: !name || name.length > 200 ? "Enter your name as on your ID (up to 200 characters)." : undefined,
      idType: !idType ? "Choose your ID type." : undefined,
      idNumber: idNumberError(idNumber),
      idFront: fileError(files.idFront, "Add the front of the ID.") ?? undefined,
      idBack: fileError(files.idBack, philsys ? "Add the back of the card." : null) ?? undefined,
      selfie: fileError(files.selfie, "Add a photo of you holding this ID.") ?? undefined,
      address: !addr || addr.length > 500 ? "Enter your address (up to 500 characters)." : undefined,
      phone: !isValidPhone(phone) ? "Enter a Philippine phone number starting with +63 or 0." : undefined,
      birForm: fileError(files.birForm, null) ?? undefined,
      biometricConsent: withLive && !consent ? CONSENT_ERROR : undefined,
    });
  };

  // Live path: validate → session → check → submit at once with its id.
  // Admin path: the same submit with no session id (the server records DECLINED).
  const send = async (withLive: boolean) => {
    if (busy) return;
    const found = validate(withLive);
    setErrors(found);
    if (Object.keys(found).length > 0) return;
    const isCurrent = captureAuthSession();
    setSending(withLive ? "live" : "admin");
    try {
      const livenessSessionId = withLive ? await live.run() : undefined;
      if (!isCurrent()) return;
      const saved = await submitSellerDetails({
        legalName: legalName.trim(),
        idType: idType as SellerIdType,
        idNumber: idNumber.trim(),
        address: address.trim(),
        phone: phone.trim(),
        idFront: files.idFront!,
        idBack: files.idBack,
        selfie: files.selfie!,
        birForm: files.birForm,
        livenessSessionId,
      });
      if (!isCurrent()) return;
      showToast({ kind: "success", message: reviewStateOf(saved) === "auto_approved" ? "Seller details approved." : "Seller details sent. We'll check them soon." });
      onSent(saved);
    } catch (err) {
      if (isCurrent()) setErrors(errorsFrom(err));
    } finally {
      if (isCurrent()) setSending(null);
    }
  };

  const textProps = (field: string) => fieldProps(field, errors);
  const fileRow = (field: FileField, label: string, hint?: string) => (
    <div className="flex flex-col gap-2">
      <label htmlFor={`seller-${field}`} className={LABEL_CLS}>{label}</label>
      {hint && <p className="font-sans text-sm text-slate-soft">{hint}</p>}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <label className={cn(BTN_SECONDARY, BTN_SIZE.sm, "cursor-pointer")}>
          {files[field] ? "Change" : "Choose photo"}
          <input
            {...textProps(field)}
            type="file"
            accept={SELLER_FILE_TYPES.join(",")}
            disabled={busy}
            onChange={(e) => { pick(field, e.target.files?.[0] ?? null); e.target.value = ""; }}
            className="sr-only"
          />
        </label>
        {files[field] && (
          <>
            <span className="min-w-0 truncate font-sans text-sm text-ink">{files[field]!.name}</span>
            <button type="button" onClick={() => pick(field, null)} disabled={busy} className={cn(BTN_GHOST, "text-sm")}>
              Remove
            </button>
          </>
        )}
      </div>
      <FieldError message={errors[field]} id={`seller-${field}-error`} density="tight" />
    </div>
  );

  return (
    <form onSubmit={(e) => { e.preventDefault(); void send(true); }} noValidate className="flex flex-col gap-7">
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-legalName" className={LABEL_CLS}>Legal name (as on your ID)</label>
        <input {...textProps("legalName")} value={legalName} maxLength={200} autoComplete="name" disabled={busy}
          onChange={(e) => { setLegalName(e.target.value); clear("legalName"); }} className={INPUT_CLS} />
        <FieldError message={errors.legalName} id="seller-legalName-error" density="tight" />
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-idType" className={LABEL_CLS}>ID type</label>
        <select {...textProps("idType")} value={idType} disabled={busy}
          onChange={(e) => { setIdType(e.target.value as SellerIdType | ""); clear("idType"); clear("idBack"); }} className={INPUT_CLS}>
          <option value="">Choose one</option>
          {SELLER_ID_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
        </select>
        <FieldError message={errors.idType} id="seller-idType-error" density="tight" />
      </div>
      <IdNumberField value={idNumber} onChange={(v) => { setIdNumber(v); clear("idNumber"); }} errors={errors} disabled={busy} />
      {fileRow("idFront", "Front of the ID")}
      {fileRow("idBack", philsys ? "Back of the card (the side with the QR code)" : "Back of the ID (optional)")}
      {fileRow("selfie", "A photo of you holding this ID", "Your face and the front of the ID should both be clear.")}
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-address" className={LABEL_CLS}>Address</label>
        <textarea {...textProps("address")} value={address} rows={2} maxLength={500} autoComplete="street-address" disabled={busy}
          onChange={(e) => { setAddress(e.target.value); clear("address"); }} className={cn(INPUT_CLS, "resize-none")} />
        <FieldError message={errors.address} id="seller-address-error" density="tight" />
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-phone" className={LABEL_CLS}>Phone number</label>
        <input {...textProps("phone")} type="tel" value={phone} maxLength={30} autoComplete="tel" placeholder="0917 123 4567" disabled={busy}
          onChange={(e) => { setPhone(e.target.value); clear("phone"); }} className={cn(INPUT_CLS, "tnum")} />
        <FieldError message={errors.phone} id="seller-phone-error" density="tight" />
      </div>
      {fileRow("birForm", "BIR Form 2303 (optional)")}
      <ConsentTick checked={consent} onChange={(v) => { setConsent(v); clear("biometricConsent"); }} errors={errors} disabled={busy} />
      <FieldError message={errors.form} id="seller-form-error" density="tight" />
      <div className="flex flex-col gap-5">
        <button type="submit" disabled={busy} className={cn(BTN_PRIMARY, BTN_SIZE.md, "w-fit")}>
          {sending === "live" ? "Checking your details…" : "Take the live selfie and send"}
        </button>
        <div className="flex flex-col gap-3">
          <p className="font-sans text-sm text-slate"><LegalText text={LEGAL_COPY.identityAdminInstead} /></p>
          <div className="flex flex-wrap items-center gap-4">
            <button type="button" onClick={() => void send(false)} disabled={busy} className={cn(BTN_SECONDARY, BTN_SIZE.sm)}>
              {sending === "admin" ? "Sending…" : "Have an admin check instead"}
            </button>
            {onCancel && (
              <button type="button" onClick={onCancel} disabled={busy} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
                Cancel
              </button>
            )}
          </div>
        </div>
      </div>
      {live.modal}
    </form>
  );
}
