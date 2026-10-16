"""What machine the service runs on, read once at startup so a speed measurement can be understood.

Railway gives the blur service a share of a host it does not name. The CPU model, its
instruction-set flags (for example VNNI, which makes 8-bit models fast) and the cgroup limits
(the vCPU quota and the memory cap) explain a per-photo time that a laptop cannot reproduce.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

CPUINFO = Path("/proc/cpuinfo")
CPU_MAX = Path("/sys/fs/cgroup/cpu.max")
MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")
INTERESTING_FLAGS = ("avx2", "fma", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8")


def host_facts() -> dict:
    """CPU count, model, the interesting flags that are present, the cgroup quota and memory cap.

    Every file is optional (Windows and macOS have none of them): a missing file leaves its
    keys out.
    """
    facts: dict = {"cpu_count": os.cpu_count()}
    try:
        for line in CPUINFO.read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "model name" and "cpu_model" not in facts:
                facts["cpu_model"] = value.strip()
            elif key == "flags" and "cpu_flags" not in facts:
                present = set(value.split())
                facts["cpu_flags"] = [flag for flag in INTERESTING_FLAGS if flag in present]
    except OSError:
        pass
    for key, path in (("cpu_max", CPU_MAX), ("memory_max", MEMORY_MAX)):
        with contextlib.suppress(OSError):
            facts[key] = path.read_text(encoding="utf-8").strip()
    return facts
