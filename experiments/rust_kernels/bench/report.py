"""Render benchmark JSON as Markdown tables.

    .venv/bin/python experiments/rust_kernels/bench/report.py [results.json]

Writes the Markdown next to the JSON (same name, ``.md``) and prints it.
Sections missing from the JSON (e.g. a ``--only huygens`` run) are skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def ms(seconds: float) -> str:
    """Milliseconds with about three significant figures."""
    value = seconds * 1e3
    if value >= 100:
        return f"{value:,.0f}"
    return f"{value:.2f}" if value < 10 else f"{value:.1f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    """A Markdown table with the first column left-aligned."""
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def timing_rows(cases: list[dict], labels: dict[str, str], baseline: str):
    """One row per implementation: time and speedup over ``baseline``."""
    rows = []
    for key, label in labels.items():
        if not any(key in case["seconds"] for case in cases):
            continue
        cells = [label]
        for case in cases:
            t, base = case["seconds"].get(key), case["seconds"][baseline]
            cells.append("-" if t is None else f"{ms(t)} ({base / t:.1f}x)")
        rows.append(cells)
    return rows


def huygens_section(cases: list[dict], th: int) -> list[str]:
    """Huygens-Fresnel summation table."""
    header = ["Implementation"] + [
        f"{c['image'][0]}x{c['image'][1]} image, {c['pupil_points']:,} pupil pts"
        for c in cases
    ]
    labels = {}
    for threads in (th, 1):
        suffix = "1 thread" if threads == 1 else f"{threads} threads"
        for key, name in (
            ("optiland_numba", "Optiland Numba (shipped)"),
            ("rust_literal", "Rust, literal port"),
            ("numba_hoisted", "Numba, hoisted phase"),
            ("rust_hoisted", "Rust, hoisted phase"),
            ("numba_simd", "Numba, hoisted + fast sincos"),
            ("rust_simd", "Rust, hoisted + fast sincos (SIMD)"),
        ):
            labels[f"{key}@{threads}t"] = f"{name}, {suffix}"
    rows = timing_rows(cases, labels, f"optiland_numba@{th}t")
    return [
        "### Huygens-Fresnel PSF summation",
        "",
        f"Baseline: Optiland's shipped Numba kernel on {th} threads.",
        "",
        table(header, rows),
    ]


def trace_section(cases: list[dict], th: int) -> list[str]:
    """One sequential-trace table per lens."""
    labels = {
        "optiland_numpy@1t": "Optiland NumPy (shipped)",
        "numba_fused@1t": "Numba fused, 1 thread",
        "rust_fused@1t": "Rust fused, 1 thread",
        f"numba_fused@{th}t": f"Numba fused, {th} threads",
        f"rust_fused@{th}t": f"Rust fused, {th} threads",
    }
    by_lens: dict[str, list] = {}
    for row in cases:
        by_lens.setdefault(row["lens"], []).append(row)
    out = [
        "### Sequential real-ray trace (surface loop only)",
        "",
        "Baseline: Optiland's NumPy surface loop (`SurfaceGroup.trace`, "
        "`record=False`).",
    ]
    for lens, rows in by_lens.items():
        header = ["Implementation"] + [f"{c['rays']:,} rays" for c in rows]
        out += [
            "",
            f"**{lens}** ({rows[0]['surfaces']} surfaces)",
            "",
            table(header, timing_rows(rows, labels, "optiland_numpy@1t")),
        ]
    return out


def fft_section(cases: list[dict], th: int) -> list[str]:
    """2-D FFT table."""
    labels = {
        "numpy_pocketfft@1t": "numpy.fft.fft2 (shipped)",
        "scipy_fft@1t": "scipy.fft.fft2, 1 worker",
        "rustfft@1t": "rustfft, 1 thread",
        f"scipy_fft@{th}t": f"scipy.fft.fft2, {th} workers",
        f"rustfft@{th}t": f"rustfft, {th} threads",
    }
    header = ["Implementation"] + [f"{c['size']}x{c['size']}" for c in cases]
    return [
        "### 2-D complex FFT",
        "",
        "Baseline: `numpy.fft.fft2` (pocketfft), used by Optiland's FFT PSF/MTF.",
        "",
        table(header, timing_rows(cases, labels, "numpy_pocketfft@1t")),
    ]


def end_to_end_section(cases: list[dict], th: int) -> list[str]:
    """Kernel-swap table."""
    rows = [
        [r["workload"], ms(r["stock"]), ms(r["rust"]), f"{r['stock'] / r['rust']:.1f}x"]
        for r in cases
    ]
    header = ["Workload", "Optiland as shipped", "With Rust kernel", "Speedup"]
    return [
        "### End to end: Rust kernel swapped into a real Optiland call",
        "",
        table(header, rows),
    ]


SECTIONS = {
    "huygens": huygens_section,
    "trace": trace_section,
    "fft": fft_section,
    "end_to_end": end_to_end_section,
}


def render(data: dict) -> str:
    """Markdown for every section present in ``data``."""
    env, th = data["environment"], data["environment"]["threads"]
    out = [
        f"Machine: {env['cpu']}, {th} threads. Python {env['python']}, "
        f"NumPy {env['numpy']}, SciPy {env['scipy']}, Numba {env['numba']} "
        f"(SVML {'on' if env.get('numba_svml') else 'off'}), {env['rustc']}. "
        f"Run {env['date']}.",
        "",
        "Cells: best wall time in ms (speedup over the baseline).",
    ]
    for key, section in SECTIONS.items():
        if key in data:
            out += ["", *section(data[key], th)]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS / "benchmarks.json"
    text = render(json.loads(path.read_text()))
    path.with_suffix(".md").write_text(text)
    print(text)
