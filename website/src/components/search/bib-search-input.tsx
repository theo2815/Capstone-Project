"use client";

import { useState, type FormEvent } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";

interface BibSearchInputProps {
  onSearch: (bibNumber: string) => void;
  isSearching?: boolean;
}

export function BibSearchInput({
  onSearch,
  isSearching,
}: BibSearchInputProps) {
  const [bib, setBib] = useState("");

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const trimmed = bib.trim();
    if (trimmed) onSearch(trimmed);
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">
        Search by Bib Number
      </h3>
      <form onSubmit={handleSubmit} className="flex gap-3">
        <Input
          id="bib"
          value={bib}
          onChange={(e) => setBib(e.target.value)}
          placeholder="Enter bib number (e.g. 1023)"
          className="max-w-xs"
        />
        <Button type="submit" isLoading={isSearching} disabled={!bib.trim()}>
          <Search className="mr-2 h-4 w-4" /> Search
        </Button>
      </form>
    </div>
  );
}
