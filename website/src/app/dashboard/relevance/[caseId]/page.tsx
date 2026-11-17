"use client";

import { useParams } from "next/navigation";
import { RelevanceCase } from "./relevance-case";

export default function RelevanceCasePage() {
  const { caseId } = useParams<{ caseId: string }>();
  return <RelevanceCase caseId={caseId} />;
}
