"use client";

import { useMemo, useState } from "react";
import {
  getDailyDecisionsSeed30d,
  getDailyUploads30d,
  totalAmount,
} from "@/lib/admin-overview-mock";
import { AdminTrendChart } from "./admin-trend-chart";
import { useAdminKpiTrend } from "@/hooks/use-admin-data";
import { cn } from "@/lib/utils";

type TrendMode = "decisions" | "uploads";

// Toggle state is local; decision totals come from the backend trend.
export function AdminOverviewTrend() {
  const [mode, setMode] = useState<TrendMode>("decisions");
  // Live-mode override for decisions: prefer GET /admin/kpis/trend?days=30.
  // Uploads stay derived (no server endpoint specced).
  const liveTrend = useAdminKpiTrend(30);

  const series = useMemo(() => {
    if (mode === "uploads") return getDailyUploads30d();
    if (liveTrend) {
      return liveTrend.map((point) => ({
        date: point.date,
        amount: point.decisions,
      }));
    }
    return getDailyDecisionsSeed30d();
  }, [mode, liveTrend]);

  const total = useMemo(() => totalAmount(series), [series]);
  const unit = mode === "decisions" ? "decisions" : "uploads";

  return (
    <div>
      <div className="flex items-center gap-2 mb-5">
        <ToggleButton
          active={mode === "decisions"}
          onClick={() => setMode("decisions")}
        >
          Decisions
        </ToggleButton>
        <ToggleButton
          active={mode === "uploads"}
          onClick={() => setMode("uploads")}
        >
          Uploads
        </ToggleButton>
        <span className="ml-auto font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] text-slate-soft tnum">
          {total.toLocaleString()} {unit} · last 30 days
        </span>
      </div>

      <AdminTrendChart
        data={series}
        unitLabel={unit}
        ariaLabel={`Daily ${unit} for the last 30 days`}
      />
    </div>
  );
}

function ToggleButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        "font-mono uppercase tracking-[0.14em] text-[14px] min-[400px]:text-[15px] md:text-[13px] rounded-full px-4 py-1.5 transition-colors",
        active
          ? "bg-fresh text-surface"
          : "border border-line text-slate hover:text-ink hover:border-ink",
      )}
    >
      {children}
    </button>
  );
}
