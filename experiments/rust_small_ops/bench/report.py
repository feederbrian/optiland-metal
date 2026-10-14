"""Render ``results/small_ops.json`` (and ``characterize.json``) as Markdown.

    .venv/bin/python experiments/rust_small_ops/bench/report.py

Writes ``results/small_ops.md`` and prints it.
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"


def fmt(seconds: float) -> str:
    """Latency with a unit that keeps three significant figures."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.0f} ns"
    if seconds < 1e-3:
        return (
            f"{seconds * 1e6:.1f} µs" if seconds < 1e-4 else f"{seconds * 1e6:.0f} µs"
        )
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:,.1f} s"


def speedup(ratio: float) -> str:
    """One decimal below 10x, whole numbers above."""
    return f"{ratio:.1f}x" if ratio < 10 else f"{ratio:,.0f}x"


def table(header: list[str], rows: list[list[str]]) -> str:
    """Markdown table, first column left-aligned."""
    align = ["---"] + ["---:"] * (len(header) - 1)
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def characterize_section(c: dict) -> list[str]:
    """Baseline costs measured by characterize.py."""
    rows = [[name, fmt(t)] for name, t in c["latency"].items()]
    scaling = [
        [f"{r['rays']:,}", fmt(r["seconds"]), fmt(r["seconds"] / r["rays"])]
        for r in c["ray_scaling"]
    ]
    out = [
        "### Baseline: Optiland's small operations (Cooke triplet)",
        "",
        table(["Operation", "Per call"], rows),
        "",
        table(["Rays traced", "Per call", "Per ray"], scaling),
        "",
    ]
    for name, prof in c["profile"].items():
        shares = [[label, f"{share:.0%}"] for label, share in prof["shares"].items()]
        out += [
            f"**Where `{name}` spends its time** "
            f"({prof['calls_per_invocation']:,.0f} Python calls per call)",
            "",
            table(["Category (self time)", "Share"], shares),
            "",
        ]
    o = c["optimize"]
    out.append(
        f"Tutorial optimization as shipped: {o['iterations']} iterations, "
        f"{o['merit_evaluations']:,} merit evaluations, {fmt(o['wall_seconds'])} "
        f"({fmt(o['seconds_per_evaluation'])} per evaluation), final merit "
        f"{o['final_merit']:.4g}."
    )
    return out


def boundary_section(b: dict) -> list[str]:
    """Fixed per-call costs."""
    return [
        "### A. Cost of one call, before any work",
        "",
        table(["Call", "Per call"], [[k, fmt(v)] for k, v in b.items()]),
    ]


def leaf_section(leaf: dict) -> list[str]:
    """Rust leaf inside Optiland."""
    ops = list(leaf["stock"])
    rows = [
        [op, fmt(leaf["stock"][op]), fmt(leaf["rust leaf"][op]),
         speedup(leaf["stock"][op] / leaf["rust leaf"][op])]
        for op in ops
    ]  # fmt: skip
    return [
        "### B. Rust leaf: surface arithmetic in Rust, Optiland orchestration as is",
        "",
        table(["Operation", "Stock", "Rust leaf", "Speedup"], rows),
    ]


def patches_section(pat: dict) -> list[str]:
    """Each Optiland prototype alone and combined."""
    base = pat["stock"]
    ops = list(base)
    rows = [
        [name] + [f"{fmt(r[op])} ({speedup(base[op] / r[op])})" for op in ops]
        for name, r in pat.items()
    ]
    return [
        "### B2. Optiland prototypes, alone and combined (results bit-identical)",
        "",
        table(["Change", *ops], rows),
    ]


def blas_section(blas: dict) -> list[str]:
    """OpenBLAS oversubscription."""
    settings = list(blas)
    names = list(blas[settings[0]])
    rows = [[name] + [fmt(blas[k][name]) for k in settings] for name in names]
    return [
        "### G. OpenBLAS threads vs parallel kernels (free-running optimization)",
        "",
        table(["Parallel-gradient run", *settings], rows),
    ]


def compiled_section(comp: dict) -> list[str]:
    """Per-operation latency for every implementation."""
    ops = list(next(iter(comp.values())))
    base = comp["Optiland (shipped)"]
    rows = []
    for name, results in comp.items():
        cells = [name]
        for op in ops:
            t = results.get(op)
            cells.append("-" if t is None else f"{fmt(t)} ({speedup(base[op] / t)})")
        rows.append(cells)
    return [
        "### C. Per-operation latency (speedup over Optiland as shipped)",
        "",
        table(["Implementation", *ops], rows),
    ]


def gradient_section(g: dict) -> list[str]:
    """One finite-difference gradient."""
    base = g["Optiland (shipped), sequential"]
    rows = [[k, fmt(v), speedup(base / v)] for k, v in g.items()]
    return [
        "### D. One finite-difference gradient (8 merit evaluations)",
        "",
        table(["Implementation", "Time", "Speedup"], rows),
    ]


def _run_rows(runs: dict, base: float) -> list[list[str]]:
    return [
        [
            name,
            fmt(r["wall_seconds"]),
            speedup(base / r["wall_seconds"]),
            f"{r['merit_evaluations']:,}",
            f"{r['rescored_merit']:.5g}",
        ]
        for name, r in runs.items()
    ]


OPTION_NAMES = {
    "lsq_optiland": "Optiland LeastSquares (trf)",
    "lsq_optiland_fixes": "Optiland LeastSquares (trf) + Python fixes",
    "lsq_rust": "Rust compiled lens, least_squares (trf)",
    "lsq_rust_batched": "Rust compiled lens, least_squares (trf), parallel Jacobian",
    "fd_processes": "Optiland L-BFGS-B, gradients over 4 processes",
    "fd_processes_fixes": "Optiland L-BFGS-B, 4 processes + Python fixes",
}


def optimize_section(opt: dict, options: dict | None) -> list[str]:
    """Replay (same work) and free-running optimization tables."""
    replay, traj = opt["replay"], opt["trajectory"]
    base = replay.get("Optiland (shipped)")
    rows = [
        [name, fmt(t), "-" if base is None else speedup(base / t)]
        for name, t in replay.items()
    ]
    out = [
        "### E. The whole tutorial optimization, same work for everyone",
        "",
        f"The {traj['merit_evaluations']:,} merit evaluations of the tutorial's "
        f"L-BFGS-B run ({traj['iterations']} iterations, final merit "
        f"{traj['final_merit']:.6g}), recorded from Optiland and replayed through "
        "each implementation; every replayed merit matched Optiland's to 1e-9. "
        "Optiland rows are full optimizer runs on that same trajectory.",
        "",
        table(["Implementation", "Wall time", "Speedup"], rows),
    ]
    within = traj.get("evaluations_to_within")
    if within:
        crow = [
            [f"within {float(f):.1%} of the final merit", f"{k:,}", f"{k / n:.0%}"]
            for f, k in within.items()
            for n in [traj["merit_evaluations"]]
        ]
        out += [
            "",
            "**How much of that run was needed.** Evaluations before the best merit "
            "so far came within a given margin of where the run finally stopped:",
            "",
            table(["Best merit so far", "Evaluations", "Share of run"], crow),
        ]
    header = ["Run", "Wall time", "Speedup", "Merit evals", "Final merit (Optiland)"]
    if base is not None and opt.get("free_running"):
        out += [
            "",
            "**Free-running optimizations.** Same start and settings, but each run "
            "follows its own path (see the trajectory-sensitivity note):",
            "",
            table(header, _run_rows(opt["free_running"], base)),
        ]
    if options and base is not None:
        runs = {OPTION_NAMES.get(k, k): v for k, v in options.items()}
        out += [
            "",
            "### F. Options that change how many evaluations run, or where",
            "",
            table(header, _run_rows(runs, base)),
        ]
    return out


def render() -> str:
    """Markdown for everything measured so far."""
    data = json.loads((RESULTS / "small_ops.json").read_text())
    env = data["environment"]
    out = [
        f"Machine: {env['cpu']}, {env['threads']} threads. Python {env['python']}, "
        f"NumPy {env['numpy']}, Numba {env['numba']}, {env['rustc']}. "
        f"Run {env['date']}.",
        "",
    ]
    characterize = RESULTS / "characterize.json"
    if characterize.exists():
        out += characterize_section(json.loads(characterize.read_text())) + [""]
    for key, fn in (
        ("boundary", boundary_section),
        ("leaf", leaf_section),
        ("patches", patches_section),
        ("compiled", compiled_section),
        ("gradient", gradient_section),
    ):
        if key in data:
            out += fn(data[key]) + [""]
    if "optimize" in data:
        out += optimize_section(data["optimize"], data.get("options"))
    if "multistart" in data:
        m = data["multistart"]
        out += [
            "",
            f"### H. Multi-start: {m['starts']} perturbed starts "
            f"(sigma {m['sigma']} in scaled units), Numba lens + parallel "
            f"gradient: {fmt(m['wall_seconds'])} "
            f"in total; best merit {m['best_rescored_by_optiland']:.8g} (Optiland "
            f"re-score; stock run {m.get('stock_final_merit', float('nan')):.8g}), "
            f"median {m['median_merit']:.4g}.",
        ]
    if "blas" in data:
        out += ["", *blas_section(data["blas"])]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    text = render()
    (RESULTS / "small_ops.md").write_text(text)
    print(text)
