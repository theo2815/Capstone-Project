"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import Link from "next/link";
import { PhotoPreviewCard, type PhotoPreviewItem } from "@/components/photos/photo-preview-card";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { usePhotographerPhotoPurchases } from "@/hooks/use-photographer-purchases";
import { ApiError } from "@/lib/api";
import { type PhotographerPurchase } from "@/lib/api-photographer-earnings";
import { fetchPhotographerPhotoDownload } from "@/lib/api-photographer";
import { captureAuthSession } from "@/lib/auth-session";
import { formatSalesDate } from "@/lib/format";
import type { PhotographerLibraryPhoto, PhotographerTransaction } from "@/lib/photographer-mock";
import { savePhoto } from "@/lib/save-photo";
import { formatOrderRef, formatPrice } from "@/lib/utils";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";

export interface PurchaseSelection {
  photoId: string;
  orderId?: string;
  transactionId?: string;
}

interface Props {
  selection: PurchaseSelection | null;
  photos?: PhotographerLibraryPhoto[];
  transaction?: PhotographerTransaction;
  onClose: () => void;
  onSelectPhoto?: (photoId: string) => void;
}

export function PhotographerPurchasePreview(props: Props) {
  const { selection, onClose } = props;
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  const sessionKey = JSON.stringify(session.key);
  const [selectionSession, setSelectionSession] = useState<string | null>(sessionKey);
  const closeNotified = useRef<string | null>(null);
  useEffect(() => {
    if (!session.enabled) setSelectionSession(null);
    else if (!selection) {
      setSelectionSession(sessionKey);
      closeNotified.current = null;
    }
    if (selection && (!session.enabled || selectionSession !== sessionKey) && closeNotified.current !== sessionKey) {
      closeNotified.current = sessionKey;
      onClose();
    }
  }, [selection, onClose, session.enabled, sessionKey, selectionSession]);
  if (!selection || !session.enabled || selectionSession !== sessionKey) return null;
  return <OpenPurchasePreview key={`${selection.photoId}:${selection.orderId ?? ""}:${selection.transactionId ?? ""}`} {...props} selection={selection} />;
}

function OpenPurchasePreview({ selection, photos, transaction, onClose, onSelectPhoto }: Props & { selection: PurchaseSelection }) {
  const selectionKey = `${selection.photoId}:${selection.orderId ?? ""}:${selection.transactionId ?? ""}`;
  const activeSelection = useRef<string | null>(selectionKey);
  useLayoutEffect(() => { activeSelection.current = selectionKey; return () => { activeSelection.current = null; }; }, [selectionKey]);
  const pinnedOrderId = selection.orderId ?? transaction?.orderId;
  const [orderId, setOrderId] = useState(pinnedOrderId);
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState(false);
  const purchase = usePhotographerPhotoPurchases(selection.photoId, pinnedOrderId, true);
  const photo = purchase.photo ?? photos?.find((item) => item.id === selection.photoId) ?? null;
  const gallery = photos?.length ? photos : photo ? [photo] : [];
  const index = gallery.findIndex((item) => item.id === selection.photoId);
  const fallback = transaction && transaction.photoId === selection.photoId && (!orderId || transaction.orderId === orderId) ? transaction : null;
  const selected = (orderId && purchase.selectedPurchase?.orderId === orderId ? purchase.selectedPurchase : null)
    ?? purchase.items.find((item) => item.orderId === orderId)
    ?? (!orderId && !fallback ? purchase.selectedPurchase ?? purchase.items[0] : null);
  const preview: PhotoPreviewItem = {
    id: selection.photoId,
    bib: photo?.bib ?? null,
    time: "",
    tone: photo?.tone ?? 0,
    price: 0,
    span: photo?.span,
    imageUrl: photo?.thumbnailUrl ?? null,
  };

  async function download() {
    if (!purchase.photo || purchase.photo.deletionStartedAt || downloading) return;
    const sameSession = captureAuthSession();
    const isCurrent = () => sameSession() && activeSelection.current === selectionKey;
    setDownloading(true);
    setDownloadError(false);
    try {
      const result = await fetchPhotographerPhotoDownload(selection.photoId);
      if (!isCurrent()) return;
      if (!result?.url) throw new Error("Download unavailable");
      await savePhoto(result.url);
    } catch {
      if (isCurrent()) setDownloadError(true);
    } finally {
      if (isCurrent()) setDownloading(false);
    }
  }

  const errorText = purchase.error instanceof ApiError
    ? purchase.error.status === 404 ? "Purchase unavailable" : purchase.error.status === 429 ? "Too many requests. Try again shortly." : "Couldn't load purchases."
    : "Could not load purchases.";

  const refundRecordedAt = selected?.transactions.filter((row) => row.isRefund).at(-1)?.recordedAt;
  const factClass = "min-w-0 break-words text-right tnum";

  const details = <div className="space-y-3 text-sm text-ink">
    <p className="font-display font-bold">Purchase details</p>
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-2">
      <dt className="text-slate">Buyer</dt><dd className={factClass}>{selected?.buyer || fallback?.buyer || "Unavailable"}</dd>
      <dt className="text-slate">Reference</dt><dd className={factClass}>{selected?.orderId || orderId ? formatOrderRef(selected?.orderId ?? orderId ?? "") : "Unavailable"}</dd>
      <dt className="text-slate">Purchased</dt><dd className={factClass}>{selected?.purchasedAt ? formatSalesDate(selected.purchasedAt) : fallback?.paidAt ? formatSalesDate(fallback.paidAt) : "Unavailable"}</dd>
      {selected && !selected.purchasedAt && <><dt className="text-slate">Order created</dt><dd className={factClass}>{formatSalesDate(selected.orderCreatedAt)}</dd></>}
      <dt className="text-slate">Event</dt><dd className={factClass}>{purchase.event?.id || transaction?.eventId
        ? <Link href={`/dashboard/events/${encodeURIComponent(purchase.event?.id ?? transaction!.eventId)}?tab=purchased`} className="underline decoration-line-strong underline-offset-4 hover:decoration-fresh">{purchase.event?.name ?? transaction?.eventName ?? "Event archived"} · Purchased photos</Link>
        : "Event archived"}</dd>
      <dt className="text-slate">Photo</dt><dd className={factClass}>{photo?.bib ? `Bib ${photo.bib}` : selection.photoId}</dd>
      <dt className="text-slate">Photo price</dt><dd className={factClass}>{selected ? formatPrice(selected.pricePhp) : "Unavailable"}</dd>
      <dt className="text-slate">Discount</dt><dd className={factClass}>{selected ? formatPrice(selected.discountPhp) : "Unavailable"}</dd>
      <dt className="text-slate">Paid for this photo</dt><dd className={factClass}>{selected ? selected.kind === "free" ? `Free (${formatPrice(selected.paidPhp)})` : formatPrice(selected.paidPhp) : "Unavailable"}</dd>
      <dt className="text-slate">Refunded</dt><dd className={factClass}>{selected?.refundedPhp == null ? "Unavailable" : formatPrice(selected.refundedPhp)}</dd>
      <dt className="text-slate">Refund status</dt><dd className={factClass}>{selected ? selected.refundStatus.charAt(0).toUpperCase() + selected.refundStatus.slice(1) : "Unavailable"}</dd>
      {selected?.refundedAt && <><dt className="text-slate">Refund completed</dt><dd className={factClass}>{formatSalesDate(selected.refundedAt)}</dd></>}
      {!selected?.refundedAt && refundRecordedAt && <><dt className="text-slate">Refund recorded</dt><dd className={factClass}>{formatSalesDate(refundRecordedAt)}</dd></>}
      <dt className="text-slate">Your net earnings</dt><dd className={factClass}>{selected?.netKeptPhp == null ? "Unavailable" : formatPrice(selected.netKeptPhp)}</dd>
    </dl>
    {selected?.refundPending && <p>Refund pending</p>}
    {selected?.transactions.map((row) => <p key={row.id} aria-current={row.id === selection.transactionId ? "true" : undefined} className={row.id === selection.transactionId ? "rounded-lg border border-ink bg-fresh-tint/40 p-2 tnum" : "p-2 tnum"}>
      {row.isRefund ? "Earnings adjustment" : "Sale earnings"} · {formatPrice(row.amountKept)} · Recorded {formatSalesDate(row.recordedAt)}
    </p>)}
    {!selected && !fallback && <p>Purchase unavailable.</p>}
    {fallback && !selected && !purchase.isLoading && <p>Purchase unavailable.</p>}
    {purchase.isLoading && <p>Loading purchases…</p>}
    {purchase.error && <p role="alert">{errorText}</p>}
    {(purchase.error || ((orderId || fallback) && !selected && !purchase.isLoading)) && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void purchase.refetch()}>Retry purchases</button>}
    {purchase.items.length > 1 && <div className="space-y-2 border-t border-line pt-3">
      <p className="font-display font-bold">Purchase history</p>
      {purchase.items.map((item: PhotographerPurchase) => <button key={`${item.orderId}:${item.photoId}`} type="button" aria-pressed={item.orderId === orderId} onClick={() => setOrderId(item.orderId)} className="block w-full rounded-lg border border-line p-2 text-left hover:bg-bone-deep aria-pressed:border-ink aria-pressed:bg-fresh-tint/30">
        {item.buyer || "Buyer unavailable"} · {item.purchasedAt ? `Purchased ${formatSalesDate(item.purchasedAt)}` : `Order created ${formatSalesDate(item.orderCreatedAt)}`}
      </button>)}
    </div>}
    {purchase.appendError && <p role="alert">Could not load more purchases.</p>}
    {purchase.hasNextPage && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void purchase.fetchNextPage()} disabled={purchase.isFetchingNextPage}>Load more purchases</button>}
    {downloadError && <p role="alert">Download unavailable. Try again.</p>}
  </div>;

  return <PhotoPreviewCard
    mode="owned"
    metadataAccess="owner"
    photo={preview}
    eventName={purchase.event?.name ?? transaction?.eventName ?? "Event archived"}
    eventDate={purchase.event?.date}
    index={index >= 0 ? index + 1 : 1}
    total={gallery.length || 1}
    onClose={onClose}
    onPrev={onSelectPhoto && index > 0 ? () => onSelectPhoto(gallery[index - 1].id) : undefined}
    onNext={onSelectPhoto && index >= 0 && index < gallery.length - 1 ? () => onSelectPhoto(gallery[index + 1].id) : undefined}
    onDownload={() => void download()}
    downloadDisabled={!purchase.photo || !!purchase.photo.deletionStartedAt || downloading}
    details={details}
    onRefreshImage={() => void purchase.refetch()}
  />;
}
