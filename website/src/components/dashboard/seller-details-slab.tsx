"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Slab } from "@/components/profile-shell/slab";
import { BTN_GHOST, BTN_PRIMARY, BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { FieldError } from "@/components/ui/field-error";
import { Skeleton } from "@/components/ui/skeleton";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { useToast } from "@/hooks/use-toast";
import { ApiError } from "@/lib/api";
import {
  getSellerDetails,
  isValidPhone,
  SELLER_FILE_MAX_BYTES,
  SELLER_FILE_TYPES,
  SELLER_ID_TYPES,
  submitSellerDetails,
  type SellerDetails,
  type SellerIdType,
} from "@/lib/api-seller-details";
import { captureAuthSession } from "@/lib/auth-session";
import { ROUTES } from "@/lib/constants";
import { cn } from "@/lib/utils";

type FileField = "idFront" | "idBack" | "selfie" | "birForm";
type Errors = Partial<Record<string, string>>;
const FIELDS = ["legalName", "idType", "idFront", "idBack", "selfie", "address", "phone", "birForm"];

const INPUT_CLS =
  "w-full bg-transparent border-b border-line focus:border-fresh focus:outline-none py-3 text-base text-ink placeholder:text-slate-soft transition-colors";
const LABEL_CLS = "font-sans text-sm text-slate";
const idLabel = (value?: string | null) => SELLER_ID_TYPES.find((t) => t.value === value)?.label ?? "";

// Settings section 09 (SC-2a): the photographer's seller details, which the
// backend needs before new uploads (SELLER_DETAILS_REQUIRED).
export function SellerDetailsSlab() {
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const queryClient = useQueryClient();
  const queryKey = ["photographer", "seller-details", ...session.key];
  const query = useQuery({
    queryKey,
    queryFn: ({ signal }) => getSellerDetails(signal),
    enabled: session.enabled,
  });
  const [resending, setResending] = useState(false);
  const details = query.data;
  const onFile = details?.status === "submitted" || details?.status === "approved";

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
        ) : onFile && !resending ? (
          <div>
            <p className="font-sans text-base font-semibold text-ink">
              {details.status === "approved" ? "Approved" : "We're checking your details. You can upload meanwhile."}
            </p>
            <dl className="mt-5 grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 font-sans text-sm">
              <dt className="text-slate">Legal name</dt><dd className="text-ink">{details.legalName}</dd>
              <dt className="text-slate">ID</dt>
              <dd className="text-ink">{idLabel(details.idType)}</dd>
              <dt className="text-slate">Address</dt><dd className="text-ink break-words">{details.address}</dd>
              <dt className="text-slate">Phone</dt><dd className="text-ink tnum">{details.phone}</dd>
              <dt className="text-slate">BIR Form 2303</dt><dd className="text-ink">{details.hasBirForm ? "Sent" : "Not sent"}</dd>
            </dl>
            <button type="button" onClick={() => setResending(true)} className={cn(BTN_SECONDARY, BTN_SIZE.sm, "mt-6")}>
              Send new details
            </button>
          </div>
        ) : (
          <>
            {details?.status === "rejected" && (
              <div role="status" className="mb-8 rounded-xl border border-error/40 bg-error/5 px-5 py-4 font-sans text-sm text-ink">
                <p className="font-semibold">Your seller details were not accepted.</p>
                <p className="mt-1">Reason: {details.rejectionReason}</p>
                <p className="mt-1">New uploads are paused until you send them again.</p>
              </div>
            )}
            <SellerDetailsForm
              initial={details}
              onCancel={resending ? () => setResending(false) : undefined}
              onSent={(saved) => {
                queryClient.setQueryData(queryKey, saved);
                setResending(false);
              }}
            />
          </>
        )}
      </div>
    </Slab>
  );
}

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
  const [legalName, setLegalName] = useState(initial?.legalName ?? "");
  const [idType, setIdType] = useState<SellerIdType | "">(initial?.idType ?? "");
  const [address, setAddress] = useState(initial?.address ?? "");
  const [phone, setPhone] = useState(initial?.phone ?? "");
  const [files, setFiles] = useState<Record<FileField, File | null>>({ idFront: null, idBack: null, selfie: null, birForm: null });
  const [errors, setErrors] = useState<Errors>({});
  const [sending, setSending] = useState(false);
  const philsys = idType === "philsys";

  const clear = (field: string) => setErrors((prev) => ({ ...prev, [field]: undefined, form: undefined }));
  const pick = (field: FileField, file: File | null) => {
    setFiles((prev) => ({ ...prev, [field]: file }));
    clear(field);
  };

  const validate = (): Errors => {
    const name = legalName.trim();
    const addr = address.trim();
    const found: Errors = {
      legalName: !name || name.length > 200 ? "Enter your name as on your ID (up to 200 characters)." : undefined,
      idType: !idType ? "Choose your ID type." : undefined,
      idFront: fileError(files.idFront, "Add the front of the ID.") ?? undefined,
      idBack: fileError(files.idBack, philsys ? "Add the back of the card." : null) ?? undefined,
      selfie: fileError(files.selfie, "Add a photo of you holding this ID.") ?? undefined,
      address: !addr || addr.length > 500 ? "Enter your address (up to 500 characters)." : undefined,
      phone: !isValidPhone(phone) ? "Enter a Philippine phone number starting with +63 or 0." : undefined,
      birForm: fileError(files.birForm, null) ?? undefined,
    };
    return Object.fromEntries(Object.entries(found).filter(([, v]) => v));
  };

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    if (sending) return;
    const found = validate();
    setErrors(found);
    if (Object.keys(found).length > 0) return;
    const isCurrent = captureAuthSession();
    setSending(true);
    try {
      const saved = await submitSellerDetails({
        legalName: legalName.trim(),
        idType: idType as SellerIdType,
        address: address.trim(),
        phone: phone.trim(),
        idFront: files.idFront!,
        idBack: files.idBack,
        selfie: files.selfie!,
        birForm: files.birForm,
      });
      if (!isCurrent()) return;
      showToast({ kind: "success", message: "Seller details sent. We'll check them soon." });
      onSent(saved);
    } catch (err) {
      if (!isCurrent()) return;
      const server: Errors = {};
      if (err instanceof ApiError) {
        for (const e of err.errors) server[e.field && FIELDS.includes(e.field) ? e.field : "form"] = e.message;
      } else {
        server.form = "We couldn't send your details. Check your connection and try again.";
      }
      setErrors(server);
    } finally {
      if (isCurrent()) setSending(false);
    }
  };

  const textProps = (field: string) => ({
    id: `seller-${field}`,
    "aria-invalid": errors[field] ? true : undefined,
    "aria-describedby": errors[field] ? `seller-${field}-error` : undefined,
  });
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
            disabled={sending}
            onChange={(e) => { pick(field, e.target.files?.[0] ?? null); e.target.value = ""; }}
            className="sr-only"
          />
        </label>
        {files[field] && (
          <>
            <span className="min-w-0 truncate font-sans text-sm text-ink">{files[field]!.name}</span>
            <button type="button" onClick={() => pick(field, null)} disabled={sending} className={cn(BTN_GHOST, "text-sm")}>
              Remove
            </button>
          </>
        )}
      </div>
      <FieldError message={errors[field]} id={`seller-${field}-error`} density="tight" />
    </div>
  );

  return (
    <form onSubmit={submit} noValidate className="flex flex-col gap-7">
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-legalName" className={LABEL_CLS}>Legal name (as on your ID)</label>
        <input {...textProps("legalName")} value={legalName} maxLength={200} autoComplete="name" disabled={sending}
          onChange={(e) => { setLegalName(e.target.value); clear("legalName"); }} className={INPUT_CLS} />
        <FieldError message={errors.legalName} id="seller-legalName-error" density="tight" />
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-idType" className={LABEL_CLS}>ID type</label>
        <select {...textProps("idType")} value={idType} disabled={sending}
          onChange={(e) => { setIdType(e.target.value as SellerIdType | ""); clear("idType"); clear("idBack"); }} className={INPUT_CLS}>
          <option value="">Choose one</option>
          {SELLER_ID_TYPES.map((t) => <option key={t.value} value={t.value}>{t.label}</option>)}
        </select>
        <FieldError message={errors.idType} id="seller-idType-error" density="tight" />
      </div>
      {fileRow("idFront", "Front of the ID")}
      {fileRow("idBack", philsys ? "Back of the card (the side with the QR code)" : "Back of the ID (optional)")}
      {fileRow("selfie", "A photo of you holding this ID", "Your face and the front of the ID should both be clear.")}
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-address" className={LABEL_CLS}>Address</label>
        <textarea {...textProps("address")} value={address} rows={2} maxLength={500} autoComplete="street-address" disabled={sending}
          onChange={(e) => { setAddress(e.target.value); clear("address"); }} className={cn(INPUT_CLS, "resize-none")} />
        <FieldError message={errors.address} id="seller-address-error" density="tight" />
      </div>
      <div className="flex flex-col gap-2">
        <label htmlFor="seller-phone" className={LABEL_CLS}>Phone number</label>
        <input {...textProps("phone")} type="tel" value={phone} maxLength={30} autoComplete="tel" placeholder="0917 123 4567" disabled={sending}
          onChange={(e) => { setPhone(e.target.value); clear("phone"); }} className={cn(INPUT_CLS, "tnum")} />
        <FieldError message={errors.phone} id="seller-phone-error" density="tight" />
      </div>
      {fileRow("birForm", "BIR Form 2303 (optional)")}
      <FieldError message={errors.form} id="seller-form-error" density="tight" />
      <div className="flex flex-wrap items-center gap-4">
        <button type="submit" disabled={sending} className={cn(BTN_PRIMARY, BTN_SIZE.md)}>
          {sending ? "Sending…" : "Send seller details"}
        </button>
        {onCancel && (
          <button type="button" onClick={onCancel} disabled={sending} className={cn(BTN_GHOST, BTN_SIZE.sm)}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
