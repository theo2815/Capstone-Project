"""The host facts line reads what exists and leaves out what does not."""

from __future__ import annotations

from src.utils import host_info


def test_reads_model_flags_and_cgroup_limits(tmp_path, monkeypatch):
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text(
        "processor\t: 0\nmodel name\t: AMD EPYC 7B13 64-Core Processor\n"
        "flags\t\t: fpu avx2 fma avx512f avx512_vnni\nprocessor\t: 1\nmodel name\t: other\n",
        encoding="utf-8",
    )
    (tmp_path / "cpu.max").write_text("200000 100000\n", encoding="utf-8")
    (tmp_path / "memory.max").write_text("2147483648\n", encoding="utf-8")
    monkeypatch.setattr(host_info, "CPUINFO", cpuinfo)
    monkeypatch.setattr(host_info, "CPU_MAX", tmp_path / "cpu.max")
    monkeypatch.setattr(host_info, "MEMORY_MAX", tmp_path / "memory.max")

    facts = host_info.host_facts()

    assert facts["cpu_model"] == "AMD EPYC 7B13 64-Core Processor"  # the first processor's
    assert facts["cpu_flags"] == ["avx2", "fma", "avx512f", "avx512_vnni"]
    assert facts["cpu_max"] == "200000 100000"  # 2 vCPUs: 200 ms of CPU per 100 ms
    assert facts["memory_max"] == "2147483648"
    assert facts["cpu_count"] is not None


def test_missing_files_leave_their_keys_out(tmp_path, monkeypatch):
    monkeypatch.setattr(host_info, "CPUINFO", tmp_path / "none")
    monkeypatch.setattr(host_info, "CPU_MAX", tmp_path / "none")
    monkeypatch.setattr(host_info, "MEMORY_MAX", tmp_path / "none")
    assert set(host_info.host_facts()) == {"cpu_count"}
