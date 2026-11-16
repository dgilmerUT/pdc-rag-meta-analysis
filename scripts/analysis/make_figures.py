"""Regenerate manuscript figures and the numbers table from outputs/descriptors.json.

Run from the repository root:  PYTHONPATH=scripts/analysis python3 scripts/analysis/make_figures.py

data/chunks.json (full text) is not distributed. Without it, the chunk count and the residual
duplicate check are read from data/chunk_counts.csv and data/residual_duplicate_groups.json.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

import stats_lib as L

OUT = Path("outputs/manuscript_v2")
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, RED = "#2c6e91", "#c85a19", "#3a7d44", "#a4243b"
BOX_FILL = "#dce8ef"
DPI = 600
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 11})

R = L.load()
P = [r for r in R if L.tier(r) == "pdc"]
PA = [r for r in R if L.tier(r) in ("pdc", "application")]
C = [r for r in R if L.tier(r) == "conventional_ceramic"]
A = [r for r in R if L.tier(r) == "application"]
tiers = {t: sum(1 for r in R if L.tier(r) == t) for t in ("pdc", "conventional_ceramic", "application", "alternative_route")}
o = L.run(P, PA)
n = o["n_pdc"]


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── Figure 1: pipeline ───────────────────────────────────────────────────────
def fig1():
    if Path("data/chunks.json").exists():
        raw_chunks = sum(1 for c in json.load(open("data/chunks.json")) if c["source"] in {r["source"] for r in R})
    else:
        srcs = {r["source"] for r in R}
        raw_chunks = sum(int(x["n_chunks"]) for x in csv.DictReader(open("data/chunk_counts.csv", encoding="utf-8"))
                         if x["source"] in srcs)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

    def rbox(x, y, w, h, fc, ec, lw=1.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(0.4, 8.5, "Curated corpus", fontsize=20, fontweight="bold", va="top")
    ax.text(0.4, 7.85, f"({len(R):,} papers)", fontsize=20, fontweight="bold", va="top")
    items = [(f"PDC primary — {tiers['pdc']:,}", 6.5), (f"Conventional ceramic — {tiers['conventional_ceramic']}", 5.5),
             (f"Application performance — {tiers['application']}", 4.5), (f"Alternative routes — {tiers['alternative_route']}", 3.5)]
    for label, y in items:
        rbox(0.3, y, 3.6, 0.7, "#FFF3E0", "#D2691E")
        ax.text(2.1, y + 0.35, label, fontsize=12, ha="center", va="center", color="#4E2C00")
        ax.annotate("", xy=(4.3, 5.3), xytext=(3.9, y + 0.35), arrowprops=dict(arrowstyle="->", color="#555555", lw=1.3))
    steps = [("Parse", "Docling PDF →\nstructured JSON", 4.3),
             ("Chunk", f"250 words,\n50-wd overlap\n{raw_chunks:,} chunks", 6.4),
             ("Embed", "SciBERT →\nChromaDB\n(HNSW, cosine)", 8.5),
             ("Retrieve", "Top-40 chunks\nper paper,\nsection-weighted", 10.6),
             ("Extract", "LLM, 11-category\nschema,\nexplicit nulls", 12.7)]
    bw = 1.9
    for title, desc, x in steps:
        rbox(x, 4.2, bw, 2.1, "#D6E4F0", "#1B4F8A")
        rbox(x, 5.6, bw, 0.7, "#2E75B6", "#1B4F8A")
        ax.text(x + bw / 2, 5.95, title, fontsize=16, fontweight="bold", ha="center", va="center", color="white", zorder=3)
        ax.text(x + bw / 2, 4.95, desc, fontsize=11, ha="center", va="center", color="#1A3A5C", linespacing=1.5, zorder=3)
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(steps[i + 1][2], 5.3), xytext=(steps[i][2] + bw, 5.3), arrowprops=dict(arrowstyle="->", color="#2E75B6", lw=2))
    rbox(4.3, 1.0, 10.3, 2.2, "#A8D5BA", "#4A8C6F", lw=2.5)
    ax.text(9.45, 2.65, "Corpus-scale descriptor landscape", fontsize=18, fontweight="bold", ha="center", va="center", color="#2D6A4F")
    ax.text(9.45, 1.75, "coverage / gap analysis  •  structure–property correlations with provenance\nPDC vs. conventional-route benchmarking",
            fontsize=13, ha="center", va="center", color="#1B3D2F", linespacing=1.6)
    ax.annotate("", xy=(9.45, 3.2), xytext=(9.45, 4.2), arrowprops=dict(arrowstyle="->", color="#4A8C6F", lw=2.5))
    fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return raw_chunks


# ── Figure 2: families and architecture ─────────────────────────────────────
def fig2():
    fam_order = [("polysiloxane", "Polysiloxane"), ("polycarbosilane", "Polycarbosilane"), ("polysilazane", "Polysilazane"),
                 ("polyborosilazane", "Polyborosilazane"), ("polysilsesquioxane", "Polysilsesquioxane"),
                 ("polysilylcarbodiimide", "Polysilylcarbodiimide"), ("multiple", "Multiple families")]
    arch_order = [("network", "Network"), ("linear", "Linear"), ("branched", "Branched"), ("hyperbranched", "Hyperbranched"),
                  ("cyclic", "Cyclic"), ("mixed", "Mixed / multiple")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, order, counts, color, title, xl in [
        (axes[0], fam_order, o["fam"], BLUE, "(a) Preceramic polymer families", f"Papers (of {o['n_typed']:,} PDC-tier papers with polymer type)"),
        (axes[1], arch_order, o["arch"], ORANGE, "(b) Backbone architecture entering pyrolysis", f"Papers (of {sum(o['arch'].values()):,} PDC-tier papers with architecture)")]:
        vals = [counts.get(k, 0) for k, _ in order]
        y = np.arange(len(order))
        ax.barh(y, vals, color=color, height=0.6)
        ax.set_yticks(y, [lab for _, lab in order]); ax.invert_yaxis()
        for yi, v in zip(y, vals):
            ax.text(v + max(vals) * 0.01, yi, str(v), va="center", fontsize=9)
        ax.set_title(title, fontweight="bold", fontsize=11); ax.set_xlabel(xl)
    fig.tight_layout(w_pad=3)
    save(fig, "fig2_landscape")


# ── Figure 3: pyrolysis windows ─────────────────────────────────────────────
def fig3():
    t, y = o["temp"]["values"], o["yield"]["values"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4))
    axes = axes.ravel()
    ax = axes[0]
    ax.hist(t, bins=np.arange(200, 2700, 100), color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(np.median(t), color=ORANGE, ls="--", lw=1.6)
    ax.text(np.median(t) + 40, ax.get_ylim()[1] * 0.97, f"median\n{np.median(t):.0f} °C", color=ORANGE, va="top", fontsize=9)
    ax.set_title(f"(a) Temperature (n = {len(t):,})", fontweight="bold", fontsize=11)
    ax.set_xlabel("Pyrolysis temperature (°C)"); ax.set_ylabel("Papers")
    ax = axes[2]
    ax.hist(y, bins=np.arange(0, 105, 5), color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(np.median(y), color=ORANGE, ls="--", lw=1.6)
    ax.text(np.median(y) - 2, ax.get_ylim()[1] * 0.97, f"median\n{np.median(y):.0f}%", color=ORANGE, va="top", ha="right", fontsize=9)
    ax.set_title(f"(c) Ceramic yield (n = {len(y)})", fontweight="bold", fontsize=11)
    ax.set_xlabel("Ceramic yield (wt%)"); ax.set_ylabel("Papers")
    ax = axes[3]
    groups = [("network", "Network"), ("mixed", "Mixed"), ("linear", "Linear"), ("not reported", "Not\nreported")]
    data = [o["yield_by_arch"].get(k, []) for k, _ in groups]
    ax.boxplot(data, widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=BOX_FILL, edgecolor=BLUE, lw=1.2), medianprops=dict(color=ORANGE, lw=2),
               whiskerprops=dict(lw=1.2), capprops=dict(lw=1.2), flierprops=dict(marker="o", ms=3, mfc="gray", mec="none"))
    ax.set_xticks(range(1, 5), [f"{lab}\n(n={len(d)})" for (_, lab), d in zip(groups, data)], fontsize=8)
    ax.set_ylabel("Ceramic yield (wt%)")
    ax.set_title("(d) Yield by architecture", fontweight="bold", fontsize=11)
    ax = axes[1]
    atm = o["atm"]
    rows = [("Argon", atm["argon"], BLUE), ("Nitrogen", atm["nitrogen"], BLUE), ("Ammonia", atm["ammonia"], BLUE),
            ("Vacuum", atm["vacuum"], BLUE), ("Air / oxidizing", atm["air"], ORANGE),
            ("Not reported", n - atm["reported"], "#b9b9b9")]
    yy = np.arange(len(rows))
    ax.barh(yy, [100 * v / n for _, v, _ in rows], color=[c for _, _, c in rows], height=0.6)
    ax.set_yticks(yy, [lab for lab, _, _ in rows]); ax.invert_yaxis()
    for yi, (_, v, _) in zip(yy, rows):
        ax.text(100 * v / n + 0.8, yi, f"{100 * v / n:.0f}% (n = {v})", va="center", fontsize=9)
    ax.set_xlim(0, 60); ax.set_xlabel(f"PDC-tier papers (%), n = {n:,}")
    ax.set_title(f"(b) Pyrolysis atmosphere (n = {atm['reported']:,} reporting)", fontweight="bold", fontsize=11)
    fig.tight_layout(w_pad=2, h_pad=2.5)
    save(fig, "fig3_pyrolysis")


# ── New figure: elemental coverage on the periodic table ───────────────────
PT_ROWS = ["H . . . . . . . . . . . . . . . . He", "Li Be . . . . . . . . . . B C N O F Ne",
           "Na Mg . . . . . . . . . . Al Si P S Cl Ar", "K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr",
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe", "Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"]
LANTH = "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
CORE = {"Si", "C", "N", "O", "B"}
ELEM_BANDS = [(21, "#16384a", "white", "> 20 papers"), (5, BLUE, "white", "5–20 papers"),
              (1, "#bcd3e2", "#222222", "1–4 papers"), (0, "#f3f3f3", "#9a9a9a", "not found")]


def fig_elements():
    counts = L.element_papers(P)
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(-0.1, 18.1); ax.set_ylim(8.4, -1.25); ax.axis("off")

    def cell(sym, col, row):
        v = counts.get(sym, 0)
        if sym in CORE:
            fc, tc, sub = ORANGE, "white", f"{v:,}"
        elif sym == "H":
            fc, tc, sub = "#f3f3f3", "#555555", ""
        else:
            fc, tc = next((f, t) for lo, f, t, _ in ELEM_BANDS if v >= lo)
            sub = str(v) if v else ""
        ax.add_patch(FancyBboxPatch((col + 0.06, row + 0.06), 0.88, 0.88, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=fc, ec="white", lw=1.5))
        ax.text(col + 0.5, row + 0.42, sym, ha="center", va="center", fontsize=11, fontweight="bold", color=tc)
        if sub:
            ax.text(col + 0.5, row + 0.76, sub, ha="center", va="center", fontsize=7.5, color=tc)

    for r, line in enumerate(PT_ROWS):
        for c, sym in enumerate(line.split()):
            if sym != ".":
                cell(sym, c, r)
    for c, sym in enumerate(LANTH):
        cell(sym, c + 3, 6.6)
    ax.text(2.9, 7.1, "Lanthanides", ha="right", va="center", fontsize=9, color="#555555")
    from matplotlib.patches import Patch
    legend = [(ORANGE, "Core PDC elements (Si, C, N, O, B)")] + [(f, lab) for lo, f, _, lab in ELEM_BANDS]
    ax.legend(handles=[Patch(fc=fc, ec="#cccccc", lw=0.6, label=lab) for fc, lab in legend], loc="upper left",
              bbox_to_anchor=(0, 1.0), ncol=5, frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.4)
    save(fig, "fig_elements")
    return {s: v for s, v in counts.items()}


# ── New figure: pyrolysis pathway with descriptor coverage per stage ───────
def fig_pathway():
    pid = sum(1 for r in P if L.reported(r.get("polymer_type")) or L.reported(r.get("polymer_name")))
    cov = {"Polymer identity": 100 * pid / n, "Molecular weight": 100 * o["cat"]["Molecular weight"] / n,
           "Crosslinking": 100 * o["cat"]["Crosslinking"] / n, "Functional groups": 100 * o["cat"]["Functional groups"] / n,
           "Thermal / pyrolysis": 100 * o["cat"]["Thermal / pyrolysis"] / n, "Ceramic yield": 100 * o["field"]["Ceramic yield"] / n,
           "Bonding": 100 * o["cat"]["Bonding environment"] / n, "Elemental comp.": 100 * o["cat"]["Elemental composition"] / n,
           "Microstructure": 100 * o["cat"]["Microstructural phase"] / n, "Mechanical props.": 100 * o["cat"]["Mechanical properties"] / n,
           "Crystallinity": 100 * o["cat"]["Crystallinity"] / n, "Grain size": 100 * o["field"]["Grain size"] / n}
    stages = [("RT–150 °C", "Polymer", "Stable, processable\nprecursor; viscosity\ncontrol", ["Polymer identity", "Molecular weight"]),
              ("150–350 °C", "Crosslinking", "Hydrosilylation,\nvinyl/allyl reactions;\ngel point", ["Crosslinking", "Functional groups"]),
              ("300–550 °C", "Organic loss", "CH₄, NH₃, H₂\nevolve; mass loss\nsets ceramic yield", ["Thermal / pyrolysis", "Ceramic yield"]),
              ("550–900 °C", "Short-range\norder", "Si tetrahedra\norder; Si–C/N/O\nbonds set; amorphous", ["Bonding", "Elemental comp."]),
              ("900–1100 °C", "Nanodomains", "β-SiC and free-C\nnanodomains (1–5 nm);\nhardness peak", ["Microstructure", "Mechanical props."]),
              ("> 1100 °C", "Long-range\norder", "Crystallization,\ngrain coarsening,\nproperty scatter", ["Crystallinity", "Grain size"])]
    shades = ["#d9e7f0", "#b3cfe0", "#7fb0cc", BLUE, "#1f5370", "#16384a"]
    fig, ax = plt.subplots(figsize=(10, 4.9))
    ax.set_xlim(0, 6); ax.set_ylim(5.05, -0.15); ax.axis("off")
    w = 0.94
    for i, ((temp, name, proc, descs), shade) in enumerate(zip(stages, shades)):
        x = i + 0.03
        dark = i >= 3
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.05, boxstyle="round,pad=0,rounding_size=0.05", fc=shade, ec="none"))
        ax.text(x + w / 2, 0.24, temp, ha="center", va="center", fontsize=9, color="white" if dark else "#16384a")
        ax.text(x + w / 2, 0.66, name, ha="center", va="center", fontsize=9.5, fontweight="bold", linespacing=1.1,
                color="white" if dark else "#16384a")
        ax.text(x + w / 2, 1.65, proc, ha="center", va="center", fontsize=8.4, color="#222222", linespacing=1.35)
        for k, dname in enumerate(descs):
            v = cov[dname]
            y0 = 2.6 + k * 1.1
            ax.text(x + 0.04, y0, dname, ha="left", va="center", fontsize=8.4, color="#222222")
            bw = w - 0.36
            ax.add_patch(Rectangle((x + 0.04, y0 + 0.22), bw, 0.3, fc="#eeeeee", ec="none"))
            ax.add_patch(Rectangle((x + 0.04, y0 + 0.22), bw * v / 100, 0.3, fc=tier_color(v), ec="none"))
            ax.text(x + w - 0.02, y0 + 0.37, f"{v:.0f}%", ha="right", va="center", fontsize=8.4, fontweight="bold", color="#222222")
        if i < 5:
            ax.annotate("", xy=(i + 1.03, 0.525), xytext=(i + 0.97, 0.525), arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1))
    ax.text(0.03, 4.85, "Bars: PDC-tier papers reporting the descriptor (green ≥ 50%, orange 20–49%, red < 20%)",
            fontsize=8.4, color="#555555", va="center")
    save(fig, "fig_pathway")
    return {k: round(v) for k, v in cov.items()}


# ── Figure 4 (new): precursor family → reported phases ──────────────────────
def fig4():
    rows = L.family_phase_matrix(P)
    M = np.array([r[2] for r in rows])
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fb", BLUE, "#16384a"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    cols = [p for p, _ in L.PHASES]
    ax.set_xticks(range(len(cols)), [{"Free C": "Free\ncarbon", "Amorphous": "Amorphous\nphase"}.get(c, c) for c in cols], fontsize=10)
    ax.set_yticks(range(len(rows)), [f"{f.capitalize()} (n={k})" for f, k, _ in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] >= 55 else "#222222")
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    expected = {"polysiloxane": "SiOC", "polycarbosilane": "SiC", "polysilazane": "SiCN",
                "polyborosilazane": "SiBCN", "polysilsesquioxane": "SiOC"}
    for i, (f, _, _) in enumerate(rows):
        j = cols.index(expected[f])
        ax.add_patch(Rectangle((j - 0.47, i - 0.45), 0.94, 0.9, fill=False, ec=ORANGE, lw=2))
    ax.set_xlabel("Papers naming the phase in extracted phase, phase-composition, or XRD fields (%)", fontsize=10)
    ax.set_title("Ceramic phases reported by single-family precursor papers", fontweight="bold", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_family_phase")
    return rows


# ── Figure 5: mechanical properties by route ────────────────────────────────
def fig5():
    cm = L.conv_mech(C)
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.9))
    for ax, key, lab, log in [(axes[0], "hardness", "Hardness (GPa)", False),
                              (axes[1], "modulus", "Elastic modulus (GPa)", False),
                              (axes[2], "strength", "Strength (MPa)", True)]:
        p, c = o["mech"][key]["values"], cm[key]
        if log:
            p, c = p[p > 0], c[c > 0]
        tf = np.log10 if log else (lambda v: v)
        for pos, vals, col in [(1, p, BLUE), (2, c, ORANGE)]:
            v = tf(np.asarray(vals, float))
            parts = ax.violinplot([v], positions=[pos], widths=0.75, showextrema=False)
            for body in parts["bodies"]:
                body.set_facecolor(col); body.set_edgecolor(col); body.set_alpha(0.22)
            ax.boxplot([v], positions=[pos], widths=0.14, whis=(0, 100), patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="white", edgecolor="#333333", lw=1), medianprops=dict(color="black", lw=1.6),
                       whiskerprops=dict(lw=1), capprops=dict(lw=0))
            ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=7, color=col, alpha=0.5, lw=0, zorder=3)
        ax.set_xticks([1, 2], [f"PDC route\n(n={len(p)})", f"Conventional\n(n={len(c)})"], fontsize=9)
        ax.set_xlim(0.4, 2.6); ax.set_ylabel(lab)
        if log:
            lo, hi = int(np.floor(tf(min(p.min(), c.min())))), int(np.ceil(tf(max(p.max(), c.max()))))
            ax.set_yticks(range(lo, hi + 1), [f"$10^{{{k}}}$" for k in range(lo, hi + 1)])
    fig.suptitle("Corpus-extracted mechanical properties by processing route", fontweight="bold", fontsize=11)
    fig.tight_layout(w_pad=3)
    save(fig, "fig5_mechanical")
    return {k: len(v) for k, v in cm.items()}


# ── Figure 6: coverage ──────────────────────────────────────────────────────
def tier_color(pct):
    return GREEN if pct >= 50 else (ORANGE if pct >= 20 else RED)


def fig6():
    cat = [(name, 100 * v / n) for name, v in o["cat"].items()]
    fld = [(name, 100 * v / n) for name, v in o["field"].items()] + [("Crosslink density (units)", 0.0)]
    fld.sort(key=lambda x: -x[1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1]})
    for ax, data, fs, title, xl in [
        (axes[0], cat, 11, "(a) Category-level coverage", f"PDC-tier papers reporting ≥1 field in category (%), n = {n:,}"),
        (axes[1], fld, 9, "(b) Individual descriptor coverage", "PDC-tier papers reporting field (%)")]:
        y = np.arange(len(data))
        cols = [tier_color(v) if ax is axes[1] else (GREEN if v >= 65 else ORANGE if v >= 40 else RED) for _, v in data]
        ax.barh(y, [v for _, v in data], color=cols, height=0.62)
        ax.set_yticks(y, [lab for lab, _ in data], fontsize=fs); ax.invert_yaxis(); ax.set_xlim(0, 100)
        for yi, (_, v) in zip(y, data):
            ax.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=fs - 1)
        ax.set_title(title, fontweight="bold", fontsize=11); ax.set_xlabel(xl)
    fig.tight_layout(w_pad=2)
    save(fig, "fig6_coverage")


if __name__ == "__main__":
    raw_chunks = fig1()
    fig2(); fig3()
    elements = fig_elements()
    pathway_cov = fig_pathway()
    fam_phase = fig4()
    conv_n = fig5()
    fig6()
    old = L.load_original()
    Po = [r for r in old if L.tier(r) == "pdc"]
    oo = L.run(Po, [r for r in old if L.tier(r) in ("pdc", "application")])
    net, lin = np.array(o["yield_by_arch"]["network"]), np.array(o["yield_by_arch"]["linear"])
    dups = (L.duplicate_records(R) if Path("data/chunks.json").exists()
            else json.load(open("data/residual_duplicate_groups.json")))
    numbers = {
        "tiers": tiers, "total": len(R), "raw_chunks": raw_chunks,
        "cat_pct": {k: round(100 * v / n) for k, v in o["cat"].items()},
        "cat_pct_original": {k: round(100 * v / oo["n_pdc"]) for k, v in oo["cat"].items()},
        "field_pct_original": {k: round(100 * v / oo["n_pdc"]) for k, v in oo["field"].items()},
        "temp_original_median": oo["temp"]["median"], "yield_original_median": oo["yield"]["median"],
        "original_survivors": sum(1 for r in old if r["source"] in {x["source"] for x in R}),
        "field_pct": {k: round(100 * v / n) for k, v in o["field"].items()},
        "fam": dict(o["fam"]), "n_typed": o["n_typed"], "arch": dict(o["arch"]),
        "temp": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in o["temp"].items() if k != "values"},
        "yield": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in o["yield"].items() if k != "values"},
        "rate": o["rate"], "atm": dict(o["atm"]),
        "mw_test": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in o["mw_test"].items()},
        "mw_test_original_p": oo["mw_test"]["p"],
        "power_shift_80": L.power_shift(net, lin, shifts=range(4, 16), sims=600),
        "xlink_method": o["xlink_method"], "xlink_thermal_only": o["xlink_thermal_only"], "xlink_uv": o["xlink_uv"],
        "xlink_density_entries": o["crosslink_density_value"],
        "mw_any": o["mw_any"], "mn": o["mn"], "pdi": o["pdi"],
        "cryst_size": o["cryst_size"], "amorphous": o["amorphous"],
        "phase_n": o["phase_n"], "phase_single": dict(o["phase_single"]),
        "porosity": o["porosity"], "mech_any": o["mech_any"],
        "mech": {k: {kk: vv for kk, vv in v.items() if kk != "values"} for k, v in o["mech"].items()},
        "conv_mech_n": conv_n, "app": L.app_cov(A),
        "family_phase": [(f, k, [round(x) for x in row]) for f, k, row in fam_phase],
        "elements": elements, "pathway_cov": pathway_cov,
        "duplicate_groups": len(dups), "duplicate_redundant_records": sum(len(g) - 1 for g in dups),
    }
    json.dump(numbers, open(OUT / "numbers.json", "w"), indent=1, default=float, sort_keys=True)
    print(json.dumps(numbers, indent=1, default=float))
