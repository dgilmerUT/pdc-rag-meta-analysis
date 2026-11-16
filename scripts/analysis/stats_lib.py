import json
import re
from collections import Counter

import numpy as np
from scipy import stats

DESC = "outputs/descriptors.json"
# the published v1 corpus is the first 949 records of the table as it stood before deduplication
ORIGINAL_DESC = "outputs/dedup_2026-10-28/descriptors_pre_dedup.json"
N_ORIGINAL = 949

CATS = [
    ("Microstructural phase", "microstructural"),
    ("Thermal / pyrolysis", "thermal_pyrolysis"),
    ("Crystallinity", "conformation_crystallinity"),
    ("Bonding environment", "bonding_structure"),
    ("Backbone architecture", "backbone_architecture"),
    ("Crosslinking", "crosslink_density"),
    ("Functional groups", "functional_groups"),
    ("Mechanical properties", "mechanical_properties"),
    ("Elemental composition", "elemental_composition"),
    ("Molecular weight", "molecular_weight"),
]

FIELDS = [
    ("Ceramic phase", ("microstructural", "ceramic_phase")),
    ("Crystallinity state", ("conformation_crystallinity", "crystallinity_state")),
    ("Pyrolysis atmosphere", ("thermal_pyrolysis", "pyrolysis_atmosphere")),
    ("Pyrolysis temperature", ("thermal_pyrolysis", "pyrolysis_temp_C")),
    ("Crosslinking method", ("crosslink_density", "crosslinking_method")),
    ("XRD phases", ("conformation_crystallinity", "xrd_phases_identified")),
    ("Backbone type", ("backbone_architecture", "type")),
    ("Hold time", ("thermal_pyrolysis", "hold_time_min")),
    ("Heating rate", ("thermal_pyrolysis", "heating_rate_C_per_min")),
    ("Ceramic yield", ("thermal_pyrolysis", "ceramic_yield_pct")),
    ("Strength", ("mechanical_properties", "strength_MPa")),
    ("Porosity", ("microstructural", "porosity_pct")),
    ("Grain size", ("microstructural", "grain_size_nm")),
    ("Elastic modulus", ("mechanical_properties", "modulus_GPa")),
    ("Crystallite size", ("conformation_crystallinity", "crystallite_size_nm")),
    ("Mw", ("molecular_weight", "Mw_g_per_mol")),
    ("Hardness", ("mechanical_properties", "hardness_GPa")),
    ("C content (at%)", ("elemental_composition", "C_atomic_pct")),
    ("Mn", ("molecular_weight", "Mn_g_per_mol")),
    ("Si content (at%)", ("elemental_composition", "Si_atomic_pct")),
    ("PDI (Đ)", ("molecular_weight", "PDI")),
]


def load():
    return json.load(open(DESC))


def load_original():
    return json.load(open(ORIGINAL_DESC))[:N_ORIGINAL]


def tier(r):
    return r.get("source_type") or "pdc"


def reported(v):
    if v is None or v is False:
        return False
    if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "n/a", "not reported"):
        return False
    return True


def get(r, *keys):
    for k in keys:
        if not isinstance(r, dict):
            return None
        r = r.get(k)
    return r


def num(v):
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def cat_cov(P, key):
    return sum(1 for r in P if isinstance(r.get(key), dict) and any(reported(v) for v in r[key].values()))


def field_cov(P, path):
    return sum(1 for r in P if reported(get(r, *path)))


def boot_median_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = np.median(rng.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(meds, [2.5, 97.5])


def boot_diff_ci(a, b, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = np.median(rng.choice(a, (n, len(a))), axis=1) - np.median(rng.choice(b, (n, len(b))), axis=1)
    return np.percentile(d, [2.5, 97.5])


def power_shift(a, b, shifts=range(0, 21), sims=1000, seed=0):
    """Smallest median shift (wt%) detectable at 80% power, resampling the pooled distribution."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate([a, b])
    for s in shifts:
        hits = 0
        for _ in range(sims):
            x = rng.choice(pool, len(a)) + s
            y = rng.choice(pool, len(b))
            if stats.mannwhitneyu(x, y, alternative="two-sided").pvalue < 0.05:
                hits += 1
        if hits / sims >= 0.8:
            return s
    return None


ARCHS = ["network", "linear", "branched", "hyperbranched", "cyclic"]


def arch_class(v):
    if not reported(v):
        return None
    s = str(v).strip().lower()
    if s in ARCHS:
        return s
    if s in ("cage", "ladder"):
        return "cyclic"
    return "mixed"


FAMILIES = ["polysiloxane", "polycarbosilane", "polysilazane", "polyborosilazane",
            "polysilsesquioxane", "polysilylcarbodiimide"]


def family_class(v):
    # exact single-family label; anything else (lists, hybrids, other names) counts as "multiple"
    if not reported(v):
        return None
    s = str(v).lower().strip().rstrip("s")
    return s if s in FAMILIES else "multiple"


PHASES = [
    ("SiOC", r"sioc|si-o-c|sico|oxycarbide"),
    ("SiC", r"(?<![a-z])(β-|beta-|α-|3c-)?sic(?![a-z])|silicon carbide"),
    ("SiCN", r"sicn|si-c-n|carbonitride"),
    ("Si₃N₄", r"si3n4|si₃n₄|silicon nitride"),
    ("SiBCN", r"sibcn|si-b-c-n|borocarbonitride"),
    ("SiO₂", r"sio2|sio₂|cristobalite|silica"),
    ("Free C", r"free carbon|turbostratic|graphit|free c\b|sp2 carbon"),
    ("Amorphous", r"amorph"),
]
MAP_FAMILIES = ["polysiloxane", "polycarbosilane", "polysilazane", "polyborosilazane", "polysilsesquioxane"]


def family_phase_matrix(P):
    rows = []
    for f in MAP_FAMILIES:
        S = [r for r in P if family_class(r.get("polymer_type")) == f
             and reported(get(r, "microstructural", "ceramic_phase"))]
        txt = [" ".join(str(x or "") for x in (
            get(r, "microstructural", "ceramic_phase"),
            get(r, "microstructural", "phase_composition_notes"),
            get(r, "conformation_crystallinity", "xrd_phases_identified"))).lower() for r in S]
        rows.append((f, len(S), [100 * sum(1 for t in txt if re.search(rx, t)) / len(S) for _, rx in PHASES]))
    return rows


def duplicate_records(R, chunks_path="data/chunks.json", frac=0.6):
    """Corpus records whose chunk text largely duplicates another record (filename variants of one paper)."""
    import hashlib
    from collections import defaultdict
    srcs = {r["source"] for r in R}
    texts = defaultdict(set)
    for c in json.load(open(chunks_path)):
        if c["source"] in srcs:
            texts[c["source"]].add(hashlib.md5(c["text"].encode()).hexdigest())
    inv = defaultdict(set)
    for s, ts in texts.items():
        for t in ts:
            inv[t].add(s)
    pairs = defaultdict(int)
    for ss in inv.values():
        if 1 < len(ss) < 5:
            ss = sorted(ss)
            for i in range(len(ss)):
                for j in range(i + 1, len(ss)):
                    pairs[(ss[i], ss[j])] += 1
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for (a, b), n in pairs.items():
        m = min(len(texts[a]), len(texts[b]))
        if m > 5 and n >= frac * m:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for s in list(parent):
        groups[find(s)].append(s)
    return [sorted(g) for g in groups.values() if len(g) > 1]


ELEMENTS = {
    "H": "hydrogen", "Li": "lithium", "Be": "beryllium", "B": "boron", "C": "carbon", "N": "nitrogen", "O": "oxygen",
    "F": "fluorine", "Na": "sodium", "Mg": "magnesium", "Al": "alumin", "Si": "silicon", "P": "phosphorus",
    "S": "sulfur", "Cl": "chlorine", "K": "potassium", "Ca": "calcium", "Sc": "scandium", "Ti": "titanium",
    "V": "vanadium", "Cr": "chromium", "Mn": "manganese", "Fe": "iron", "Co": "cobalt", "Ni": "nickel",
    "Cu": "copper", "Zn": "zinc", "Ga": "gallium", "Ge": "germanium", "As": "arsenic", "Se": "selenium",
    "Rb": "rubidium", "Sr": "strontium", "Y": "yttri", "Zr": "zirconi", "Nb": "niobium", "Mo": "molybdenum",
    "Ru": "ruthenium", "Rh": "rhodium", "Pd": "palladium", "Ag": "silver", "Cd": "cadmium", "In": "indium",
    "Sn": "tin", "Sb": "antimony", "Te": "tellurium", "Cs": "cesium", "Ba": "barium", "La": "lanthanum",
    "Ce": "cerium", "Nd": "neodymium", "Sm": "samarium", "Gd": "gadolinium", "Er": "erbium", "Yb": "ytterbium",
    "Hf": "hafnium", "Ta": "tantalum", "W": "tungsten", "Re": "rhenium", "Os": "osmium", "Ir": "iridium",
    "Pt": "platinum", "Au": "gold", "Pb": "lead", "Bi": "bismuth",
}
# whole-word names only where the name is also an English word or a common prefix
_NAME_RX = {s: re.compile(rf"\b{n}" + (r"\b" if s in ("Sn", "Pb", "Fe", "Ag", "Au", "Cu") else ""), re.I)
            for s, n in ELEMENTS.items()}
_NAME_RX.update({"Sn": re.compile(r"\btin\b|\bstannic|\bstannous"), "Pb": re.compile(r"\blead (?:oxide|titanate|zirconate)", re.I),
                 "Fe": re.compile(r"\biron\b|\bferr", re.I), "Cu": re.compile(r"\bcopper\b|\bcupr", re.I)})
_FORMULA = re.compile(r"(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?|[xyz])?|[()\-])+")
_SYMBOL = re.compile(r"([A-Z][a-z]?)")
# acronyms and words that parse as element symbols
_NOT_FORMULA = {"UV", "PS", "PU", "PC", "PP", "IR", "HV", "In", "As", "At", "No", "Be", "He", "Co", "Ca", "La", "Pa",
                "Am", "Es", "Ho", "Ni", "Sc", "Ga", "Po", "Re", "Te", "Ce", "Pr", "Nd", "Pd", "Ti"}


UPPER_COMPOUNDS = {"BN", "WC", "VC", "VN"}


def _symbols(text, standalone):
    found = set()
    for tok in re.findall(r"[A-Za-z0-9.()]+", text.replace("-", " ")):
        tok = tok.strip("().")
        if not tok or not _FORMULA.fullmatch(tok) or tok in _NOT_FORMULA:
            continue
        syms = _SYMBOL.findall(tok)
        if not all(s in ELEMENTS for s in syms):
            continue
        letters = re.sub(r"[^A-Za-z]", "", tok)
        if letters.isupper() and len(letters) > 1 and tok not in UPPER_COMPOUNDS:
            # acronyms (PCS, HF), sample codes (PSN2, NCP200); keep true formulas such as Y2O3, B4C, CO2
            if not any(ch.isdigit() for ch in tok) or re.search(r"[A-Z]{3}|\d{3}", tok) or re.search(r"[A-Z]{2}\d+$", tok) and len(letters) > 2:
                continue
        if len(set(syms)) >= 2:
            found |= set(syms)
        elif standalone and tok == syms[0]:
            found.add(syms[0])  # bare symbol listed in the other-elements field ("B present", "Ti, Zr")
    return found


def element_papers(P):
    """Papers naming each element in the extracted composition, polymer, phase or XRD fields, or the title."""
    counts = Counter()
    for r in P:
        other = str(get(r, "elemental_composition", "other_elements") or "")
        rest = " ".join(str(x or "") for x in (
            r.get("polymer_name"), r.get("polymer_type"), get(r, "microstructural", "ceramic_phase"),
            get(r, "microstructural", "phase_composition_notes"), get(r, "conformation_crystallinity", "xrd_phases_identified"),
            r.get("source", "").replace("_", " ")))
        els = _symbols(other, standalone=True) | _symbols(rest, standalone=False)
        # "Ti", "Co", "Ni" and similar are listed alone in the elements field; elsewhere require a formula or name
        els |= {s for s in ("Ti", "Co", "Ni", "La", "Ce", "Nd", "Pd", "Ga", "Sc", "Re", "In") if re.search(rf"\b{s}\b", other)}
        els |= {s for s, rx in _NAME_RX.items() if rx.search(other + " " + rest)}
        counts.update(els)
    return counts


def atm_counts(P):
    c = Counter()
    for r in P:
        a = get(r, "thermal_pyrolysis", "pyrolysis_atmosphere")
        if not reported(a):
            continue
        s = str(a).lower()
        if "argon" in s or re.search(r"\bar\b", s):
            c["argon"] += 1
        if "nitrogen" in s or re.search(r"\bn2\b|\bn₂\b", s):
            c["nitrogen"] += 1
        if "ammonia" in s or "nh3" in s or "nh₃" in s:
            c["ammonia"] += 1
        if "vacuum" in s:
            c["vacuum"] += 1
        if re.search(r"\bair\b|oxidi[sz]ing|\bo2\b|oxygen", s):
            c["air"] += 1
        c["reported"] += 1
    return c


PHASE_SINGLE = {"sioc": "SiOC", "sic": "SiC", "β-sic": "β-SiC", "beta-sic": "β-SiC", "sicn": "SiCN",
                "amorphous": "amorphous", "si3n4": "Si₃N₄", "si₃n₄": "Si₃N₄"}


def run(P, PA, label=""):
    n = len(P)
    out = {"n_pdc": n}
    out["cat"] = {name: cat_cov(P, key) for name, key in CATS}
    out["field"] = {name: field_cov(P, path) for name, path in FIELDS}
    out["crosslink_density_value"] = field_cov(P, ("crosslink_density", "value"))

    t = [num(get(r, "thermal_pyrolysis", "pyrolysis_temp_C")) for r in P]
    t = np.array([x for x in t if x is not None])
    out["temp"] = dict(n=len(t), median=float(np.median(t)), min=t.min(), max=t.max(),
                       ci=boot_median_ci(t), n_exact_1200=int((t == 1200).sum()),
                       n800_1200=int(((t >= 800) & (t <= 1200)).sum()),
                       n1400_1800=int(((t >= 1400) & (t <= 1800)).sum()), values=t)

    y = [num(get(r, "thermal_pyrolysis", "ceramic_yield_pct")) for r in P]
    y = np.array([x for x in y if x is not None])
    out["yield"] = dict(n=len(y), median=float(np.median(y)), min=y.min(), max=y.max(),
                        ci=boot_median_ci(y), values=y)

    hr = [num(get(r, "thermal_pyrolysis", "heating_rate_C_per_min")) for r in P]
    hr = np.array([x for x in hr if x is not None])
    out["rate"] = dict(n=len(hr), median=float(np.median(hr)), min=hr.min(), max=hr.max(),
                       q1=float(np.percentile(hr, 25)), q3=float(np.percentile(hr, 75)))
    out["atm"] = atm_counts(P)

    archs = Counter(arch_class(get(r, "backbone_architecture", "type")) for r in P)
    archs.pop(None, None)
    out["arch"] = archs
    by_arch = {}
    for r in P:
        a = arch_class(get(r, "backbone_architecture", "type"))
        yv = num(get(r, "thermal_pyrolysis", "ceramic_yield_pct"))
        if yv is not None:
            by_arch.setdefault(a or "not reported", []).append(yv)
    out["yield_by_arch"] = by_arch
    net, lin = np.array(by_arch.get("network", [])), np.array(by_arch.get("linear", []))
    out["mw_test"] = dict(n_net=len(net), n_lin=len(lin), med_net=float(np.median(net)),
                          med_lin=float(np.median(lin)),
                          p=stats.mannwhitneyu(net, lin, alternative="two-sided").pvalue,
                          diff_ci=boot_diff_ci(net, lin))

    fams = Counter(family_class(r.get("polymer_type")) for r in P)
    fams.pop(None, None)
    out["fam"] = fams
    out["n_typed"] = sum(fams.values())

    out["xlink_method"] = field_cov(P, ("crosslink_density", "crosslinking_method"))
    out["xlink_thermal_only"] = sum(
        1 for r in P if str(get(r, "crosslink_density", "crosslinking_method") or "").strip().lower() == "thermal")
    out["xlink_uv"] = sum(
        1 for r in P if str(get(r, "crosslink_density", "crosslinking_method") or "").strip().lower() in ("uv", "uv-initiated", "photo", "uv curing"))

    out["mw_any"] = cat_cov(P, "molecular_weight")
    mn = np.array([x for x in (num(get(r, "molecular_weight", "Mn_g_per_mol")) for r in P) if x is not None])
    out["mn"] = dict(n=len(mn), min=mn.min(), max=mn.max(), n700_2500=int(((mn >= 700) & (mn <= 2500)).sum()),
                     n_gt50k=int((mn > 50000).sum()))
    pdi = np.array([x for x in (num(get(r, "molecular_weight", "PDI")) for r in P) if x is not None])
    out["pdi"] = dict(n=len(pdi), min=pdi.min(), max=pdi.max())

    cs = np.array([x for x in (num(get(r, "conformation_crystallinity", "crystallite_size_nm")) for r in P) if x is not None])
    out["cryst_size"] = dict(n=len(cs), median=float(np.median(cs)), p90=float(np.percentile(cs, 90)), max=cs.max())
    out["amorphous"] = sum(1 for r in P if "amorph" in str(get(r, "conformation_crystallinity", "crystallinity_state") or "").lower())

    ph = [get(r, "microstructural", "ceramic_phase") for r in P]
    ph = [str(x).strip() for x in ph if reported(x)]
    out["phase_n"] = len(ph)
    out["phase_single"] = Counter(PHASE_SINGLE[p.lower()] for p in ph if p.lower() in PHASE_SINGLE)

    por = np.array([x for x in (num(get(r, "microstructural", "porosity_pct")) for r in P) if x is not None])
    out["porosity"] = dict(n=len(por), median=float(np.median(por)), n_le40=int((por <= 40).sum()),
                           n_60_95=int(((por >= 60) & (por <= 95)).sum()))

    out["mech_any"] = cat_cov(P, "mechanical_properties")
    mech = {}
    for key, f in [("hardness", "hardness_GPa"), ("modulus", "modulus_GPa"), ("strength", "strength_MPa")]:
        v = np.array([x for x in (num(get(r, "mechanical_properties", f)) for r in PA) if x is not None])
        mech[key] = dict(n=len(v), median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                         q3=float(np.percentile(v, 75)), values=v)
    out["mech"] = mech
    return out


def _test_modes(text, kind):
    t = text.lower()
    if kind == "strength":
        rules = [("flexural", r"flexur|bend|3-point|4-point|three-point|four-point|modulus of rupture|ring-on-ring|biaxial"),
                 ("tensile", r"tensile|tension"), ("compressive", r"compress|crush")]
    else:
        rules = [("nanoindentation", r"nanoindent|nano-indent|berkovich"), ("Vickers/Knoop", r"vickers|\bhv\b|micro-?hardness|knoop")]
    return {name for name, rx in rules if re.search(rx, t)}


def mech_by_method(PA):
    """Strength and hardness values grouped by the test method named in the extracted free-text mechanical notes."""
    out = {}
    for key, field in [("strength", "strength_MPa"), ("hardness", "hardness_GPa")]:
        groups = {}
        for r in PA:
            v = num(get(r, "mechanical_properties", field))
            if v is None:
                continue
            m = _test_modes(str(get(r, "mechanical_properties", "other_mechanical") or ""), key)
            k = next(iter(m)) if len(m) == 1 else ("multiple" if m else "not stated")
            groups.setdefault(k, []).append(v)
        out[key] = {k: dict(n=len(v), median=float(np.median(v))) for k, v in groups.items()}
    return out


def conv_mech(C):
    m = {}
    for key, f in [("hardness", "hardness_GPa"), ("modulus", "modulus_GPa"), ("strength", "strength_MPa")]:
        m[key] = np.array([x for x in (num(get(r, "mechanical_properties", f)) for r in C) if x is not None])
    return m


def app_cov(A):
    return dict(n=len(A),
                polymer=sum(1 for r in A if reported(r.get("polymer_type")) or reported(r.get("polymer_name"))),
                polymer_type=sum(1 for r in A if reported(r.get("polymer_type"))),
                mw=cat_cov(A, "molecular_weight"))
