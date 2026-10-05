"use client";

import { useState } from "react";
import { PhotoPreviewCard, type PhotoPreviewItem } from "@/components/photos/photo-preview-card";
import { usePrivateQuerySession } from "@/hooks/use-private-query-session";
import { usePhotographerPhotoPurchases } from "@/hooks/use-photographer-purchases";
import { ApiError } from "@/lib/api";
import { type PhotographerPurchase } from "@/lib/api-photographer-earnings";
import { fetchPhotographerPhotoDownload } from "@/lib/api-photographer";
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
  const session = usePrivateQuerySession("PHOTOGRAPHER");
  if (!props.selection || !session.enabled) return null;
  return <OpenPurchasePreview key={`${props.selection.photoId}:${props.selection.orderId ?? ""}:${props.selection.transactionId ?? ""}`} {...props} selection={props.selection} />;
}

function OpenPurchasePreview({ selection, photos, transaction, onClose, onSelectPhoto }: Props & { selection: PurchaseSelection }) {
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
    setDownloading(true);
    setDownloadError(false);
    try {
      const result = await fetchPhotographerPhotoDownload(selection.photoId);
      if (!result?.url) throw new Error("Download unavailable");
      await savePhoto(result.url);
    } catch {
      setDownloadError(true);
    } finally {
      setDownloading(false);
    }
  }

  const errorText = purchase.error instanceof ApiError
    ? purchase.error.status === 404 ? "Purchase unavailable" : purchase.error.status === 429 ? "Too many requests. Try again shortly." : "Couldn't load purchases."
    : "Could not load purchases.";

  const details = <div className="space-y-3 text-sm text-ink">
    <p className="font-display font-bold">Paid for this photo</p>
    {(selected || fallback) ? <>
      {(selected?.orderId ?? orderId) && <p className="font-mono tnum">{formatOrderRef(selected?.orderId ?? orderId ?? "")}</p>}
      <p>{selected?.buyer || fallback?.buyer || "Buyer unavailable"}</p>
      <p className="text-slate">{selected ? formatSalesDate(selected.purchasedAt ?? selected.orderCreatedAt) : fallback ? formatSalesDate(fallback.paidAt) : ""}</p>
      {selected && <>
        <p className="tnum">Paid {formatPrice(selected.paidPhp)} · Kept {selected.netKeptPhp == null ? "—" : formatPrice(selected.netKeptPhp)}</p>
        {selected.refundStatus !== "none" && <p>Refund: {selected.refundStatus}</p>}
        {selected.refundPending && <p>Refund pending</p>}
        {selected.transactions.map((row) => <p key={row.id} aria-current={row.id === selection.transactionId ? "true" : undefined} className={row.id === selection.transactionId ? "rounded-lg border border-ink bg-fresh-tint/40 p-2" : "p-2"}>
          {row.isRefund ? "Refund" : "Sale"} · {formatPrice(row.amountKept)} · {formatSalesDate(row.recordedAt)}
        </p>)}
      </>}
    </> : <p>Purchase unavailable.</p>}
    {fallback && !selected && !purchase.isLoading && <p>Purchase unavailable.</p>}
    {purchase.isLoading && <p>Loading purchases…</p>}
    {purchase.error && <p role="alert">{errorText}</p>}
    {(purchase.error || ((orderId || fallback) && !selected && !purchase.isLoading)) && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void purchase.refetch()}>Retry purchases</button>}
    {purchase.items.length > 1 && <div className="space-y-2 border-t border-line pt-3">
      <p className="font-display font-bold">Purchase history</p>
      {purchase.items.map((item: PhotographerPurchase) => <button key={`${item.orderId}:${item.photoId}`} type="button" aria-pressed={item.orderId === orderId} onClick={() => setOrderId(item.orderId)} className="block w-full rounded-lg border border-line p-2 text-left hover:bg-bone-deep aria-pressed:border-ink aria-pressed:bg-fresh-tint/30">
        {item.buyer || "Buyer unavailable"} · {formatSalesDate(item.purchasedAt ?? item.orderCreatedAt)}
      </button>)}
    </div>}
    {purchase.appendError && <p role="alert">Could not load more purchases.</p>}
    {purchase.hasNextPage && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void purchase.fetchNextPage()} disabled={purchase.isFetchingNextPage}>Load more purchases</button>}
    {!preview.imageUrl && <button type="button" className={`${BTN_SECONDARY} ${BTN_SIZE.sm}`} onClick={() => void purchase.refetch()}>Refresh preview</button>}
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
  />;
}
