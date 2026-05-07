"use client";

import { useEffect, useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import { AdminTextInput } from "@/components/admin/admin-form-fields";

interface AdminEditPhotographerModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (patch: { handle: string | null; brandName: string | null }) => void;
  initialHandle: string | null;
  initialBrandName: string | null;
}

// Force-edit modal. Admin can override a photographer's handle / brand name
// when normal self-serve flows aren't enough (e.g. handle squatting, brand
// name dispute). Each field-level change is logged separately so the
// activity timeline reads as discrete decisions.
export function AdminEditPhotographerModal({
  open,
  onClose,
  onSubmit,
  initialHandle,
  initialBrandName,
}: AdminEditPhotographerModalProps) {
  const [handle, setHandle] = useState(initialHandle ?? "");
  const [brandName, setBrandName] = useState(initialBrandName ?? "");

  // Re-sync if the underlying photographer changes while the modal is mounted
  // (e.g. admin opens a different row).
  useEffect(() => {
    if (open) {
      setHandle(initialHandle ?? "");
      setBrandName(initialBrandName ?? "");
    }
  }, [open, initialHandle, initialBrandName]);

  function handleSubmit() {
    const nextHandle = handle.trim().toLowerCase();
    const nextBrand = brandName.trim();
    onSubmit({
      handle: nextHandle.length > 0 ? nextHandle : null,
      brandName: nextBrand.length > 0 ? nextBrand : null,
    });
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Force-edit profile"
      intro="Override the photographer's handle or brand name. Each change is logged with the previous value for the audit trail."
      submitLabel="Save changes"
    >
      <AdminTextInput
        id="edit-handle"
        label="Public handle"
        value={handle}
        onChange={setHandle}
        prefix="@"
        placeholder="paksitphotos"
        sanitize={(raw) => raw.replace(/[^a-z0-9_-]/gi, "")}
      />
      <AdminTextInput
        id="edit-brand"
        label="Brand name"
        value={brandName}
        onChange={setBrandName}
        maxLength={60}
        showCounter
        placeholder="Paksit Photos"
      />
    </AdminFormModal>
  );
}
