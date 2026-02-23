"""Build script for the _eventai_cpp C++ extension.

Usage:
    python build_cpp.py          # Build the extension
    python build_cpp.py --clean  # Remove build artifacts and rebuild

The compiled extension (.pyd on Windows, .so on Linux) is copied
to the project root so it can be imported directly.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BUILD_DIR = PROJECT_ROOT / "build" / "cpp"
EXT_SUFFIX = sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"
MODULE_NAME = f"_eventai_cpp{EXT_SUFFIX}"


def find_vcvarsall() -> str | None:
    """Find vcvarsall.bat for MSVC on Windows."""
    search_paths = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
        Path(r"C:\Program Files\Microsoft Visual Studio"),
    ]
    for base in search_paths:
        if not base.exists():
            continue
        for vcvars in base.rglob("vcvarsall.bat"):
            return str(vcvars)
    return None


def build() -> None:
    """Configure and build the C++ extension with CMake."""
    clean = "--clean" in sys.argv

    if clean and BUILD_DIR.exists():
        print(f"Cleaning {BUILD_DIR}")
        shutil.rmtree(BUILD_DIR)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Find pybind11 cmake config via the installed package
    try:
        import pybind11

        pybind11_dir = pybind11.get_cmake_dir()
    except ImportError:
        print("ERROR: pybind11 not installed. Run: pip install pybind11", file=sys.stderr)
        sys.exit(1)

    # Determine generator and build environment
    cmake_args = [
        "cmake",
        str(PROJECT_ROOT),
        f"-B{BUILD_DIR}",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-Dpybind11_DIR={pybind11_dir}",
    ]

    env = os.environ.copy()

    if platform.system() == "Windows":
        # Use Ninja if available (faster), else MSVC default
        ninja_path = shutil.which("ninja")
        if ninja_path:
            cmake_args.extend(["-G", "Ninja"])
        else:
            # Let CMake pick the Visual Studio generator
            pass

        # Find and activate MSVC environment
        vcvars = find_vcvarsall()
        if vcvars:
            # Run vcvarsall and capture the environment
            cmd = f'"{vcvars}" x64 >nul 2>&1 && set'
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if "=" in line:
                        key, _, value = line.partition("=")
                        env[key] = value
    else:
        # Unix: prefer Ninja if available
        if shutil.which("ninja"):
            cmake_args.extend(["-G", "Ninja"])

    # Configure
    print("=" * 60)
    print("Configuring CMake...")
    print(f"  Build dir: {BUILD_DIR}")
    print(f"  Extension: {MODULE_NAME}")
    print("=" * 60)

    subprocess.run(cmake_args, env=env, check=True)

    # Build
    print("\nBuilding...")
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR), "--config", "Release"],
        env=env,
        check=True,
    )

    # Find and copy the built artifact to project root
    artifact = _find_artifact()
    if artifact is None:
        print("\nERROR: Could not find built extension!", file=sys.stderr)
        sys.exit(1)

    dest = PROJECT_ROOT / MODULE_NAME
    shutil.copy2(artifact, dest)
    print(f"\nCopied {artifact.name} -> {dest}")
    print(f"\nVerify: python -c \"import _eventai_cpp; print(_eventai_cpp.__version__)\"")


def _find_artifact() -> Path | None:
    """Search the build directory for the compiled extension."""
    for path in BUILD_DIR.rglob(f"_eventai_cpp*"):
        if path.suffix in (".pyd", ".so"):
            return path
    return None


if __name__ == "__main__":
    build()
