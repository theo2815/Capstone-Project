"use client";

import { useEffect, useId, useState } from "react";
import { Dropdown, DropdownItem } from "./dropdown";
import { FieldError } from "./field-error";
import { Kicker } from "./kicker";
import { getRegion, REGION_GROUP_LABEL, type PHRegion, type RegionGroup } from "@/lib/ph-regions";
import { cn } from "@/lib/utils";

export interface RegionProvinceValue {
  regionCode: string | null;
  provinceCode: string | null;
}

interface RegionProvinceFieldsProps {
  value: RegionProvinceValue;
  onChange: (value: RegionProvinceValue) => void;
  regions: readonly PHRegion[];
  disabled?: boolean;
  errors?: { region?: string | null; province?: string | null };
  panelClassName?: string;
  onOpenChange?: (open: boolean) => void;
}

const GROUPS: readonly RegionGroup[] = ["luzon", "visayas", "mindanao"];

export function RegionProvinceFields({ value, onChange, regions, disabled = false, errors, panelClassName, onOpenChange }: RegionProvinceFieldsProps) {
  const id = useId();
  const region = value.regionCode ? getRegion(regions, value.regionCode) : undefined;
  const province = region?.provinces.find((p) => p.code === value.provinceCode);
  const [regionOpen, setRegionOpen] = useState(false);
  const [provinceOpen, setProvinceOpen] = useState(false);
  useEffect(() => { onOpenChange?.(regionOpen || provinceOpen); }, [regionOpen, provinceOpen, onOpenChange]);
  useEffect(() => () => onOpenChange?.(false), [onOpenChange]);

  const regionLabel = region?.name ?? value.regionCode ?? "Pick a region";
  const provinceLabel = province?.name ?? value.provinceCode ?? (region ? "Pick a province" : "Pick a region first");
  return (
    <div className="grid md:grid-cols-2 gap-5 md:gap-7">
      {(["region", "province"] as const).map((field) => {
        const isRegion = field === "region";
        const label = isRegion ? "Region" : "Province";
        const selected = isRegion ? regionLabel : provinceLabel;
        const error = errors?.[field];
        return (
          <div key={field} className="flex min-w-0 flex-col gap-2">
            <label htmlFor={`${id}-${field}`} className="font-sans text-sm text-slate">{label}</label>
            <Dropdown
              id={`${id}-${field}`}
              ariaLabel={`${label}: ${selected}`}
              ariaDescribedBy={error ? `${id}-${field}-error` : undefined}
              disabled={disabled || (!isRegion && !region)}
              onOpenChange={isRegion ? setRegionOpen : setProvinceOpen}
              className="w-full min-w-0 [&>button]:w-full"
              panelClassName={cn("max-h-80 overflow-y-auto w-[min(28rem,calc(100vw-2rem))]", panelClassName)}
              trigger={<span className="flex items-center justify-between gap-2 w-full border-b border-line py-4 text-left transition-colors hover:border-slate"><span className="truncate text-lg text-ink">{selected}</span><span aria-hidden="true" className="text-slate shrink-0">▾</span></span>}
            >
              {isRegion ? GROUPS.map((group) => (
                <div key={group} className="pt-2 first:pt-0">
                  <Kicker as="p" tone="soft" className="px-4 py-2">{REGION_GROUP_LABEL[group]}</Kicker>
                  {regions.filter((r) => r.group === group).map((r) => <DropdownItem className="[&>span:first-child]:whitespace-normal" key={r.code} disabled={disabled} active={value.regionCode === r.code} onClick={() => { if (!disabled && value.regionCode !== r.code) onChange({ regionCode: r.code, provinceCode: null }); }}>{r.name}</DropdownItem>)}
                </div>
              )) : region?.provinces.map((p) => <DropdownItem className="[&>span:first-child]:whitespace-normal" key={p.code} disabled={disabled} active={value.provinceCode === p.code} onClick={() => { if (!disabled && value.provinceCode !== p.code) onChange({ regionCode: value.regionCode, provinceCode: p.code }); }}>{p.name}</DropdownItem>)}
            </Dropdown>
            <FieldError message={error} id={`${id}-${field}-error`} density="tight" />
          </div>
        );
      })}
    </div>
  );
}
