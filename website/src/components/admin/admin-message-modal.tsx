"use client";

import { useState } from "react";
import { AdminFormModal } from "@/components/admin/admin-form-modal";
import {
  AdminTextInput,
  AdminTextarea,
} from "@/components/admin/admin-form-fields";

interface AdminMessageModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: () => void;
  photographerName: string;
}

// Send-message modal — UI theater for Phase 2a. There's no message queue
// backend yet. On submit, the parent shows a "Saved to outbox · backend
// queue ships in Phase B7" toast and the modal closes. The audit log does
// NOT record the message attempt (until backend persistence exists).
export function AdminMessageModal({
  open,
  onClose,
  onSubmit,
  photographerName,
}: AdminMessageModalProps) {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");

  const subjectValid = subject.trim().length > 0;
  const bodyValid = body.trim().length > 0;
  const canSend = subjectValid && bodyValid;

  function handleSubmit() {
    if (!canSend) return;
    onSubmit();
    setSubject("");
    setBody("");
  }

  return (
    <AdminFormModal
      open={open}
      onClose={onClose}
      onSubmit={handleSubmit}
      title="Send message"
      intro={`Reach out to ${photographerName} privately. The message lands in their dashboard inbox once the messaging service is online.`}
      submitLabel="Send to outbox"
      submitDisabled={!canSend}
    >
      <AdminTextInput
        id="msg-subject"
        label="Subject"
        value={subject}
        onChange={setSubject}
        maxLength={80}
        showCounter
        placeholder="Quick question about your latest upload"
      />
      <AdminTextarea
        id="msg-body"
        label="Message"
        value={body}
        onChange={setBody}
        maxLength={600}
        rows={5}
        placeholder="Hey, just wanted to flag…"
      />
    </AdminFormModal>
  );
}
