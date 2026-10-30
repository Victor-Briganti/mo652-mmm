import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# A negrito global faz o matplotlib procurar o peso bold nas fontes de math
# text, que não o têm; o aviso é inofensivo, pois nenhum gráfico usa math text.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]

PARALLEL = ["mpi", "mpi_blas"]
COLORS = {"serial": "#6b6b6b", "mpi": "#2a78d6", "mpi_blas": "#eb6834"}
LABELS = {"serial": "Serial", "mpi": "MPI", "mpi_blas": "MPI+OpenBLAS"}
MARKERS = {"mpi": "o", "mpi_blas": "D"}
REF = "#8c8c8c"
INK = "#222222"


# ===------------------------------------------------------------------------===
# Utilitários de Dados
# ===------------------------------------------------------------------------===


def load_data(path):
    df = pd.read_csv(path)
    versions = set(df["version"])
    for version in ["serial", *PARALLEL]:
        if version not in versions:
            raise SystemExit(f"{path} has no rows for version '{version}'")

    df = df.drop(columns=["run"])
    df["time_s"] = df["time_ns"] / 1e9
    df["cores"] = df["procs"] * df["threads"]
    return df


def aggregate(df):
    keys = ["version", "N", "procs", "threads", "tile", "cores"]
    agg = df.groupby(keys, as_index=False).agg(
        t=("time_s", "median"),
        t_min=("time_s", "min"),
        t_max=("time_s", "max"),
        max_rel=("max_rel", "first"),
    )

    serial = agg[agg["version"] == "serial"].set_index("N")["t"]
    agg["t_serial"] = agg["N"].map(serial)
    agg["speedup"] = agg["t_serial"] / agg["t"]
    agg["speedup_lo"] = agg["t_serial"] / agg["t_max"]
    agg["speedup_hi"] = agg["t_serial"] / agg["t_min"]
    agg["efficiency"] = agg["speedup"] / agg["cores"]

    # Eficiência relativa a versão mais rápida de 1 núcleo.
    one = agg[agg["cores"] == 1].groupby(["version", "N"])["t"].min()
    t_one = agg.join(one.rename("t_one"), on=["version", "N"])["t_one"]
    agg["rel_eff"] = t_one / (agg["cores"] * agg["t"])
    return agg


def best_configs(agg):
    idx = agg.groupby(["version", "N", "cores"])["t"].idxmin()
    best = agg.loc[idx].sort_values(["version", "N", "cores"])
    return best.reset_index(drop=True)


# ===------------------------------------------------------------------------===
# Style
# ===------------------------------------------------------------------------===


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 17,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "axes.titlesize": 20,
            "axes.titleweight": "bold",
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "axes.edgecolor": "#555555",
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.7,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "legend.fontsize": 16,
            "legend.frameon": False,
            "figure.titlesize": 24,
            "figure.titleweight": "bold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "lines.linewidth": 2,
            "lines.markersize": 7,
        }
    )


def cores_axis(ax, label="Núcleos (ranks x threads)"):
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel(label)


def config(row):
    procs = row["procs"]
    text = f"{procs} rank" if procs == 1 else f"{procs} ranks"
    if row["version"] == "mpi":
        text += f", tile {row['tile']}"
    else:
        text += f" x {row['threads']} thr"
    return text


def scatter_executions(ax, agg, n, metric):
    for v in PARALLEL:
        d = agg[(agg["version"] == v) & (agg["N"] == n)]
        ax.scatter(
            d["cores"],
            d[metric],
            s=16,
            color=COLORS[v],
            alpha=0.3,
            lw=0,
            zorder=1,
            label=f"{LABELS[v]}: outras execuções",
        )


# ===------------------------------------------------------------------------===
# Grafos de Desempenho
# ===------------------------------------------------------------------------===


def fig_speedup(best, agg):
    sizes = sorted(best["N"].unique())
    fig, axes = plt.subplots(1, len(sizes), figsize=(5 * len(sizes), 5), sharey=True)

    for ax, n in zip(axes, sizes):
        ideal = [1, 1024]
        ax.plot(
            ideal,
            ideal,
            color=REF,
            ls="--",
            lw=1.3,
            label="Speedup ideal (y = x)",
        )
        ax.axhline(1, color=COLORS["serial"], lw=1.2, label="Baseline (x1)")

        scatter_executions(ax, agg, n, "speedup")
        for v in PARALLEL:
            d = best[(best["version"] == v) & (best["N"] == n)]
            ax.fill_between(
                d["cores"],
                d["speedup_lo"],
                d["speedup_hi"],
                color=COLORS[v],
                alpha=0.18,
                lw=0,
            )
            ax.plot(
                d["cores"],
                d["speedup"],
                marker=MARKERS[v],
                color=COLORS[v],
                label=LABELS[v],
            )

            top = d.loc[d["speedup"].idxmax()]
            peak = (top["cores"], top["speedup"])
            ax.scatter(
                *peak, s=230, facecolor="none", edgecolor=COLORS[v], lw=1.8, zorder=4
            )
            speedup = top["speedup"]
            text = (
                f"Speedup {speedup:.0f}x"
                if speedup >= 10
                else f"Speedup {speedup:.1f}x"
            )
            text += f"\n{top['cores']} núcleos ({config(top)})"

            if v == "mpi":
                xytext, coords, ha = (0.03, 0.04), "axes fraction", "left"
            else:
                xytext, coords = (0, 24), "offset points"
                ha = "right" if top["cores"] >= 256 else "center"
            ax.annotate(
                text,
                xy=peak,
                xytext=xytext,
                textcoords=coords,
                ha=ha,
                va="bottom",
                fontsize=13,
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.2",
                    "fc": "white",
                    "ec": COLORS[v],
                    "lw": 1,
                    "alpha": 0.9,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": COLORS[v],
                    "lw": 1.2,
                    "shrinkB": 8,
                },
            )

        cores_axis(ax)
        ax.set_yscale("log")
        ax.set_ylim(0.1, 10000)
        ax.set_title(f"n = {n}")
    axes[0].set_ylabel("Speedup (log)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("Speedup em Relação a Serial")
    fig.tight_layout()
    return fig


def fig_efficiency(best, agg):
    sizes = sorted(best["N"].unique())
    fig, axes = plt.subplots(
        2, len(sizes), figsize=(5 * len(sizes), 8), sharex=True, sharey="row"
    )

    for col, n in enumerate(sizes):
        for ax, metric in ((axes[0, col], "efficiency"), (axes[1, col], "rel_eff")):
            ax.axhline(1, color=REF, ls="--", lw=1.3)
            scatter_executions(ax, agg, n, metric)
            for v in PARALLEL:
                d = best[(best["version"] == v) & (best["N"] == n)]
                ax.plot(
                    d["cores"],
                    d[metric],
                    marker=MARKERS[v],
                    color=COLORS[v],
                    label=LABELS[v],
                )
            ax.yaxis.set_major_formatter("{x:.0%}")

        axes[0, col].set_title(f"n = {n}")
        cores_axis(axes[1, col])

    axes[0, 0].set_ylim(0, agg["efficiency"].max() * 1.1)
    axes[1, 0].set_ylim(0, max(1.45, agg["rel_eff"].max() * 1.1))
    axes[0, 0].set_ylabel("vs. Serial")
    axes[1, 0].set_ylabel("vs. 1 Núcleo")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=REF, ls="--", lw=1.3))
    labels.append("100% eficiência")
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Eficiência por Entrada e Configuração")
    fig.tight_layout()
    return fig


# ===------------------------------------------------------------------------===
# Grafos de Erro Númerico
# ===------------------------------------------------------------------------===


def fig_error_by_version(agg):
    sizes = sorted(agg["N"].unique())
    versions = ["mpi", "mpi_blas", "serial"]

    fig, axes = plt.subplots(
        1, len(sizes), figsize=(4.6 * len(sizes), 4.8), sharey=True
    )
    for ax, n in zip(axes, sizes):
        for x, v in enumerate(versions):
            err = agg[(agg["version"] == v) & (agg["N"] == n)]["max_rel"]
            value = err.median() if v == "mpi_blas" else err.max()
            ax.bar(x, value, 0.6, color=COLORS[v], zorder=2)
            ax.text(
                x,
                value,
                f"{value:.1E}",
                ha="center",
                va="bottom",
                fontsize=14,
                color=INK,
            )
        ax.set_xticks(range(len(versions)), labels=[LABELS[v] for v in versions])
        ax.tick_params(axis="x", length=0, labelsize=13)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"n = {n}")
    top = agg[agg["version"] != "mpi_blas"]["max_rel"].max()
    axes[0].set_ylim(0, top * 1.15)
    axes[0].yaxis.set_major_formatter(lambda v, _: "0" if v == 0 else f"{v:.0E}")
    axes[0].set_ylabel("Erro relativo máximo")

    fig.suptitle("Erro Máximo por Configuração")
    fig.tight_layout()
    return fig


# ===------------------------------------------------------------------------===
# Main
# ===------------------------------------------------------------------------===


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates the benchmark figures for the Serial, MPI "
        "and MPI + OpenBLAS versions."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "paralela" / "results.csv",
        help="benchmark CSV written by run.sh",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "graphs",
        help="directory where the figures are saved",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "all"],
        default="png",
        help="output file format",
    )
    return parser.parse_args()


def print_summary(best, agg):
    serial = agg[agg["version"] == "serial"].set_index("N")["t"]
    times = []
    for n, t in serial.items():
        times.append(f"n={n}: {t:.2f} s" if t >= 0.1 else f"n={n}: {t * 1e3:.0f} ms")
    print("Serial median time: " + ", ".join(times))

    for v in PARALLEL:
        d = best[best["version"] == v]
        for n, g in d.groupby("N"):
            top = g.loc[g["speedup"].idxmax()]
            speedup = top["speedup"]
            text = f"{LABELS[v]:22s} n={n}: best "
            text += f"{speedup:.0f}x" if speedup >= 10 else f"{speedup:.1f}x"
            text += f" at {top['cores']} cores ({config(top)})"
            print(text)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    df = load_data(args.csv)
    agg = aggregate(df)
    best = best_configs(agg)

    figures = {
        "speedup_vs_processes": fig_speedup(best, agg),
        "efficiency_vs_cores": fig_efficiency(best, agg),
        "error_by_version": fig_error_by_version(agg),
    }
    exts = ["png", "pdf"] if args.format == "all" else [args.format]
    for name, fig in figures.items():
        for ext in exts:
            fig.savefig(args.out / f"{name}.{ext}", bbox_inches="tight")
        plt.close(fig)

    print_summary(best, agg)


if __name__ == "__main__":
    main()
