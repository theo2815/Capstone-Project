"use client";

import { useMemo, useState } from "react";
import { Kicker, KICKER_SIZE_CLASS } from "@/components/ui/kicker";
import { AdminTrendChart } from "./admin-trend-chart";
import { useAdminKpiTrendQuery } from "@/hooks/use-admin-data";
import { Skeleton } from "@/components/ui/skeleton";
import { BTN_SECONDARY, BTN_SIZE } from "@/components/ui/button-styles";
import { cn } from "@/lib/utils";

type TrendMode = "decisions" | "uploads";

// Decisions and uploads both come from GET /admin/kpis/trend?days=30.
// A backend without the `uploads` field (before AX-11) hides the toggle.
export function AdminOverviewTrend() {
  const [mode, setMode] = useState<TrendMode>("decisions");
  const trendQuery = useAdminKpiTrendQuery(30);
  const trend = trendQuery.data;
  const hasUploads = !!trend?.some((p) => p.uploads != null);
  const unit: TrendMode = hasUploads ? mode : "decisions";

  const series = useMemo(
    () =>
      (trend ?? []).map((point) => ({
        date: point.date,
        amount: unit === "uploads" ? point.uploads ?? 0 : point.decisions,
      })),
    [trend, unit],
  );
  const total = series.reduce((sum, d) => sum + d.amount, 0);
  const noActivity = !!trend?.every((p) => p.decisions === 0 && !p.uploads);

  if (!trend && trendQuery.isError) {
    return (
      <div role="alert" className="space-y-3">
        <p className="font-sans text-sm text-error">
          Couldn&apos;t load the trend. Please try again.
        </p>
        <button
          type="button"
          onClick={() => void trendQuery.refetch()}
          disabled={trendQuery.isFetching}
          className={cn(BTN_SECONDARY, BTN_SIZE.sm)}
        >
          Retry
        </button>
      </div>
    );
  }
  if (!trend) {
    return (
      <div aria-busy="true">
        <Skeleton className="h-20 md:h-24 w-full rounded-md" />
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-5">
        {hasUploads && (
          <>
            <ToggleButton
              active={unit === "decisions"}
              onClick={() => setMode("decisions")}
            >
              Decisions
            </ToggleButton>
            <ToggleButton
              active={unit === "uploads"}
              onClick={() => setMode("uploads")}
            >
              Uploads
            </ToggleButton>
          </>
        )}
        <Kicker tone="soft" className="ml-auto tnum">
          {total.toLocaleString()} {unit} · last 30 days
        </Kicker>
      </div>

      {noActivity ? (
        <p className="font-sans text-sm text-ink-soft">
          No activity in the last 30 days.
        </p>
      ) : (
        <AdminTrendChart
          data={series}
          unitLabel={unit}
          ariaLabel={`Daily ${unit} for the last 30 days`}
        />
      )}
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
        `min-h-[44px] font-mono uppercase ${KICKER_SIZE_CLASS.sm} rounded-full px-4 py-1.5 transition-colors`,
        active
          ? "bg-ink text-surface"
          : "border border-line text-slate hover:text-ink hover:border-ink",
      )}
    >
      {children}
    </button>
  );
}
