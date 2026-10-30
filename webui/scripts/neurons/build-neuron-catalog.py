"""Build the /neurons explorer's static data from the neuron-lens catalog (Qwen3-8B).

    uv run --no-project --with pyarrow --with numpy --with tokenizers python -I \
        scripts/neurons/build-neuron-catalog.py <catalog dir> <tokenizer.json> <out dir> [labels.jsonl]

The catalog dir holds reads/, acts/, drivers/, legible/ and heads/ (the stdlogits/neuron-lens dataset).
Per layer it keeps only the neurons whose readings are most legible, and bakes in everything the page
would otherwise compute: decoded tokens, the hub filter, merged variants, strength percentiles, which
reads to trust at that depth, and the tokens that two lenses (or a lens and a measurement) agree on.

Output: index.json (layers, curated neurons, heads), search.json (word -> poles) and L00.json-L35.json.
"""
import bisect
import collections
import json
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

KEEP_PER_LAYER = 40
LAYERS = 36
MLP = 12288

catalog, tokjson, out = sys.argv[1:4]
labels_path = sys.argv[4] if len(sys.argv) > 4 else None
cedict_path = sys.argv[5] if len(sys.argv) > 5 else None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cedict_gloss  # noqa: E402
CEDICT = cedict_gloss.load(cedict_path) if cedict_path else {}
glossary = {}


def collect_cjk(obj):
    """Every displayed string with Chinese characters gets a gloss in glossary.json."""
    if isinstance(obj, str):
        if CEDICT and obj not in glossary and cedict_gloss.CJK.search(obj):
            g = cedict_gloss.gloss(obj, CEDICT)
            glossary[obj] = list(g) if g else None
    elif isinstance(obj, dict):
        for v in obj.values():
            collect_cjk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            collect_cjk(v)
os.makedirs(out, exist_ok=True)

tok = Tokenizer.from_file(tokjson)
VOCAB = tok.get_vocab_size()


def decode(i):
    s = tok.decode([i])
    if "�" in s or s == "":
        return "\u0001" + (tok.id_to_token(i) or f"<{i}>")  # byte piece: shown raw and marked
    return s


TOKS = [decode(i) for i in range(VOCAB)]
BYTE = np.array([t.startswith("\u0001") for t in TOKS])


def norm(t):
    return t.strip().lower().lstrip("_-.").lstrip("/([")


def stem(t):
    """Variant key: case, leading punctuation and common English suffixes removed (release, Released, -releases)."""
    if t.startswith("\u0001"):
        return t
    k = norm(t)
    for suf in ("ing", "ed", "es", "s", "ly"):
        if k.endswith(suf) and len(k) - len(suf) >= 4:
            k = k[: -len(suf)]
            break
    return k[:-1] if k.endswith("e") and len(k) > 4 else k


STEM = [stem(t) for t in TOKS]


def is_junk(i):
    """Byte pieces, whitespace and bare punctuation: real tokens, but they don't name a concept."""
    t = TOKS[i]
    if BYTE[i]:
        return True
    s = t.strip()
    return not s or all(unicodedata.category(c)[0] in "PSZC" for c in s)


JUNK = np.array([is_junk(i) for i in range(VOCAB)])

READ_COLS = ["read_S", "read_S_z", "read_R_dclm", "read_R_dclm_z", "write_S", "write_S_z", "write_R_dclm", "write_R_dclm_z"]
COMPARE_COLS = ["write_J", "write_logit"]
SCORE = {"read_S": "val", "read_S_z": "val", "read_R_dclm": "p", "read_R_dclm_z": "val",
         "write_S": "val", "write_S_z": "val", "write_R_dclm": "p", "write_R_dclm_z": "val",
         "write_J": "p", "write_logit": "p"}
FAMILY = {c: c.removesuffix("_z") for c in READ_COLS + COMPARE_COLS}


def band(L):
    if L == 0: return "embed"
    if L <= 7: return "early"
    if L == 8: return "crossover"
    if L <= 12: return "concept"
    if L <= 20: return "dark"
    if L <= 31: return "predict"
    return "suppress"


def plan(L):
    """(input reads, output reads) to trust at this depth, primary first (the brief's §4 tables)."""
    if L <= 4: ins = ["read_S", "read_S_z"]
    elif L <= 7: ins = ["read_S_z", "read_S"]
    elif L == 8: ins = ["read_S", "read_S_z", "read_R_dclm", "read_R_dclm_z"]
    elif L <= 12: ins = ["read_R_dclm", "read_R_dclm_z"]
    elif L <= 20: ins = ["read_R_dclm_z", "read_R_dclm"]
    else: ins = ["read_R_dclm", "read_R_dclm_z", "read_S_z"]
    if L <= 4: outs = ["write_S_z", "write_R_dclm_z"]
    elif L <= 8: outs = ["write_R_dclm", "write_R_dclm_z", "write_S_z"]
    elif L <= 20: outs = ["write_R_dclm", "write_R_dclm_z"]
    else: outs = ["write_R_dclm", "write_R_dclm_z"]
    return ins, outs


def merged(ids, skip=frozenset(), limit=64):
    """Tokens in rank order with variants folded into their first occurrence: [token, count, rank]."""
    order, count = [], collections.Counter()
    first = {}
    for r, i in enumerate(ids[:limit]):
        k = STEM[i] or TOKS[i]
        if k in skip:
            continue
        if k not in count:
            first[k] = [TOKS[i], 0, r]
            order.append(k)
        count[k] += 1
    for k in order:
        first[k][1] = count[k]
    return [first[k] for k in order], order


def pct_of(sorted_vals, v):
    return round(100 * bisect.bisect_left(sorted_vals, v) / len(sorted_vals))


def parse_peaks(s):
    out = []
    for part in (s or "").split(" | "):
        m = re.match(r"^(['\"])(.*)\1×(\d+)$", part.strip(" "), re.S)
        if m:
            out.append([m.group(2).encode("utf-8").decode("unicode_escape", "ignore") if "\\" in m.group(2) else m.group(2), int(m.group(3))])
    return out


def parse_contexts(s):
    """'pre⟦hit⟧post ‖ …' -> [[pre, hit, post], …]."""
    out = []
    for part in (s or "").split(" ‖ "):
        m = re.match(r"^(.*?)⟦(.*?)⟧(.*)$", part, re.S)
        if m:
            out.append([m.group(1), m.group(2), m.group(3)])
    return out


# ---------- labels (the repo's Sonnet descriptions, §9)
labels = {}
if labels_path and os.path.exists(labels_path):
    for line in open(labels_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        m = re.match(r"L(\d+) n(\d+) ([+−-])", r["id"])
        if not m:
            continue
        key = (int(m.group(1)), int(m.group(2)), 1 if m.group(3) == "+" else -1)
        best = max(r.get("labels") or [{"label": "", "weight": 0}], key=lambda x: x.get("weight", 0))
        labels[key] = {"text": best["label"], "confidence": r.get("confidence", 0), "relation": r.get("relation"),
                       "note": r.get("note", ""), "sides": r.get("sides", {}), "source": "Claude Sonnet, from full cards (neuron-lens §9)"}


def table(kind, L, columns=None):
    p = f"{catalog}/{kind}/L{L:02d}.parquet"
    if not os.path.exists(p):
        return None
    return pq.read_table(p, columns=columns)


def arr(t, c):
    return np.array(t.column(c).to_pylist())


layers_meta, neurons_index, search = [], [], collections.defaultdict(list)
search_display = {}

ONLY = {int(x) for x in os.environ.get("NEURON_LAYERS", "").split(",") if x}
for L in range(LAYERS):
    if ONLY and L not in ONLY:
        continue
    ins, outs = plan(L)
    trusted = ins + outs
    cols_needed = ["neuron", "pole", "w_norm", "cos_ab", "null_share", "vocab_skew", "vocab_kurt"]
    for c in READ_COLS + COMPARE_COLS:
        cols_needed += [c + "_ids", f"{c}_{SCORE[c]}"]
    reads = table("reads", L, cols_needed)
    acts = table("acts", L, ["neuron", "pole", "act_top", "act_max", "rate", "rate_dclm", "rate_zh", "rate_code", "rate_chat", "pos_share", "peak_tokens", "contexts"])
    drivers = table("drivers", L, ["neuron", "pole", "drivers_ids", "drivers_val", "drivers_sum_ids", "drivers_sum_val", "s_hits", "embed_hits"])
    legible = table("legible", L)
    if reads is None:
        print(f"L{L:02d}: no reads, skipped", flush=True)
        layers_meta.append({"layer": L, "band": band(L), "missing": True})
        continue
    n = reads.num_rows
    neuron = arr(reads, "neuron"); pole = arr(reads, "pole")
    row_of = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(neuron, pole))}

    def align(t):
        """Row index into t for every reads row (tables may be ordered differently)."""
        if t is None:
            return None
        tn, tp = arr(t, "neuron"), arr(t, "pole")
        m = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(tn, tp))}
        return np.array([m.get((int(a), int(b)), -1) for a, b in zip(neuron, pole)])

    ai, di, li = align(acts), align(drivers), align(legible)
    ids = {c: np.array(reads.column(c + "_ids").to_pylist(), dtype=np.int32) for c in READ_COLS + COMPARE_COLS}
    vals = {c: np.array(reads.column(f"{c}_{SCORE[c]}").to_pylist(), dtype=np.float32) for c in READ_COLS + COMPARE_COLS}
    strength = {c: np.sort(vals[c][:, :8].mean(1)).tolist() for c in READ_COLS + COMPARE_COLS}
    hubs = {}
    for c in READ_COLS + COMPARE_COLS:
        if c.endswith("_z"):
            hubs[c] = np.zeros(VOCAB, bool)
            continue
        cnt = np.zeros(VOCAB, np.int64)
        for row in ids[c]:
            cnt[np.unique(row)] += 1
        hubs[c] = cnt / n >= 0.03
    kurt_sorted = np.sort(arr(reads, "vocab_kurt")).tolist()

    act_top = np.array(acts.column("act_top").to_pylist())[ai] if acts is not None else np.ones(n)
    silent = act_top < 0.05
    leg_cols = legible.column_names if legible is not None else []
    q_cols = [c for c in leg_cols if c.endswith("_q")]

    def leg(c, default=np.nan):
        if legible is None or c not in leg_cols:
            return np.full(n, default)
        return np.array(legible.column(c).to_pylist(), dtype=float)[li]

    side_q = {c: leg(c) for c in q_cols}
    cur_mode, next_mode = leg("cur_mode", 0.0), leg("next_mode", 0.0)
    legible_sides = np.zeros(n)
    for c, q in side_q.items():
        legible_sides += np.nan_to_num(q, nan=1.0) <= 0.05
    # findings §15: a token or next-token detector puts a third or more of its top firings on one token.
    detector = (np.nan_to_num(cur_mode) >= 1 / 3) | (np.nan_to_num(next_mode) >= 1 / 3)
    dark = (~silent) & (legible_sides == 0) & (~detector)

    # Measured evidence per pole: stems of its exact drivers and its peak tokens.
    drv_ids = np.array(drivers.column("drivers_ids").to_pylist(), dtype=np.int32)[di] if drivers is not None else None
    peaks_col = acts.column("peak_tokens").to_pylist() if acts is not None else None

    def measured_stems(i):
        s = set()
        if drv_ids is not None:
            s |= {STEM[t] for t in drv_ids[i][:20] if not JUNK[t]}
        if peaks_col is not None:
            s |= {stem(t) for t, _ in parse_peaks(peaks_col[ai[i]])}
        s.discard("")
        return s

    # ---------- selection score for every pole of the layer
    score = np.full(n, -1e9)
    for i in range(n):
        if silent[i]:
            continue
        fams = collections.defaultdict(set)
        junk8, pcts = 0, []
        for c in trusted:
            row = [t for t in ids[c][i] if not hubs[c][t]][:16]
            fams[FAMILY[c]] |= {STEM[t] for t in row if not JUNK[t]}
            pcts.append(pct_of(strength[c], float(vals[c][i, :8].mean())))
        prim = [t for t in ids[ins[0]][i] if not hubs[ins[0]][t]][:8] + [t for t in ids[outs[0]][i] if not hubs[outs[0]][t]][:8]
        junk8 = sum(JUNK[t] for t in prim) / max(1, len(prim))
        fam_list = list(fams.values())
        cross = len(set().union(*[a & b for x, a in enumerate(fam_list) for b in fam_list[x + 1:]])) if len(fam_list) > 1 else 0
        meas = measured_stems(i)
        conf = len(meas & set().union(*fam_list)) if fam_list else 0
        s = 2.0 * legible_sides[i] + 0.5 * min(conf, 6) + 0.4 * min(cross, 5) + max(pcts) / 50 + (1.0 if detector[i] else 0) - 3.0 * junk8
        s += 0.3 * min(math.log1p(float(act_top[i])), 2.5)
        key = (L, int(neuron[i]), int(pole[i]))
        if key in labels and labels[key]["confidence"] >= 0.5:
            s += 4.0
        score[i] = s
    by_neuron = collections.defaultdict(list)
    for i in range(n):
        by_neuron[int(neuron[i])].append(score[i])
    neuron_score = {k: max(v) + 0.25 * min(v) for k, v in by_neuron.items() if max(v) > -1e8}
    chosen = sorted(neuron_score, key=lambda k: -neuron_score[k])[:KEEP_PER_LAYER]
    for (lL, ln, lp), lab in labels.items():
        if lL == L and lab["confidence"] >= 0.6 and ln not in chosen:
            chosen.append(ln)

    # ---------- layer statistics over all 24,576 poles
    stats = {"silent": float(silent.mean()), "detector": float((detector & ~silent).mean()),
             "legible": float(((legible_sides > 0) & ~silent).mean()), "dark": float(dark.mean())} if legible is not None else None
    hub_list = []
    for c in ins[:1] + outs[:1]:
        if not c.endswith("_z"):
            top = np.argsort(-hubs[c].astype(int))[: int(hubs[c].sum())]
            hub_list += [TOKS[t] for t in top[:12]]

    # ---------- per neuron detail
    detail = []
    for nrn in chosen:
        entry = {"n": nrn, "poles": []}
        for pl in (1, -1):
            i = row_of[(nrn, pl)]
            fam_tokens = collections.defaultdict(set)
            rd = []
            meas = measured_stems(i)
            twins_seen = {}
            ordered = trusted + [c for c in READ_COLS if c not in trusted] + COMPARE_COLS
            for c in ordered:
                trust = "primary" if c in (ins[0], outs[0]) else "trusted" if c in trusted else "compare" if c in COMPARE_COLS else "weak"
                row = [int(t) for t in ids[c][i] if not hubs[c][t]]
                twin = twins_seen.get(FAMILY[c]) if c not in COMPARE_COLS else None
                limit = 64 if trust in ("primary", "trusted") else 24
                toks, keys = merged(row, skip=frozenset(twin or ()), limit=limit)
                if c not in COMPARE_COLS:
                    twins_seen.setdefault(FAMILY[c], set()).update(keys)
                if trust in ("primary", "trusted"):
                    fam_tokens[FAMILY[c]] |= {k for k in keys[:24] if k and not k.startswith("\u0001")}
                rd.append({"col": c, "trust": trust, "twin": bool(twin), "pct": pct_of(strength[c], float(vals[c][i, :8].mean())),
                           "tokens": toks, "hubs": int(sum(hubs[c][t] for t in ids[c][i]))})
            # agreement: a token is "shared" when another lens family names it, "measured" when it drives or peaks the pole
            fams = list(fam_tokens.items())
            shared = set()
            for x, (fa, a) in enumerate(fams):
                for fb, b in fams[x + 1:]:
                    shared |= a & b
            for r_ in rd:
                for t in r_["tokens"]:
                    k = stem(t[0])
                    t.append((1 if k in shared else 0) | (2 if k in meas else 0))
            ins_s = set().union(*[fam_tokens[FAMILY[c]] for c in ins if FAMILY[c] in fam_tokens])
            outs_s = set().union(*[fam_tokens[FAMILY[c]] for c in outs if FAMILY[c] in fam_tokens])
            p = {"pole": pl, "reads": rd, "shared": sorted(shared)[:24], "inout": sorted((ins_s & outs_s) - {""})[:16],
                 "measuredAgree": sorted(meas & (ins_s | outs_s))[:16],
                 "w_norm": round(float(reads.column("w_norm")[i].as_py()), 3), "null_share": round(float(reads.column("null_share")[i].as_py()), 4)}
            if acts is not None and ai[i] >= 0:
                a = acts.slice(int(ai[i]), 1).to_pylist()[0]
                p.update({"act_top": round(a["act_top"], 3), "act_max": round(a["act_max"], 3), "rate": a["rate"],
                          "slices": [a["rate_dclm"], a["rate_zh"], a["rate_code"], a["rate_chat"]],
                          "peaks": parse_peaks(a["peak_tokens"])[:6], "contexts": parse_contexts(a["contexts"])})
            p["silent"] = bool(silent[i])
            if drivers is not None and di[i] >= 0:
                d = drivers.slice(int(di[i]), 1).to_pylist()[0]
                p["drivers"] = [[TOKS[t], round(v, 2)] for t, v in zip(d["drivers_ids"][:24], d["drivers_val"][:24])]
                p["driversSum"] = [[TOKS[t], round(v, 2)] for t, v in zip(d["drivers_sum_ids"][:16], d["drivers_sum_val"][:16])]
                p["s_hits"] = d["s_hits"]
            p["legible"] = {c.removesuffix("_q"): round(float(side_q[c][i]), 4) for c in q_cols if not math.isnan(side_q[c][i])}
            p["detector"] = "token" if cur_mode[i] >= 1 / 3 else "next" if next_mode[i] >= 1 / 3 else None
            p["modes"] = [round(float(np.nan_to_num(cur_mode[i])), 3), round(float(np.nan_to_num(next_mode[i])), 3)]
            if L >= 21:
                kp = pct_of(kurt_sorted, float(reads.column("vocab_kurt")[i].as_py()))
                skew = float(reads.column("vocab_skew")[i].as_py())
                p["shape"] = {"kurtPct": kp, "skew": round(skew, 3), "kind": ("predicts" if skew > 0 else "suppresses") if kp >= 90 else None}
            lab = labels.get((L, nrn, pl))
            if lab:
                p["label"] = lab
            p["score"] = round(float(score[i]), 2)
            entry["poles"].append(p)
        detail.append(entry)

    # ---------- index row and search forms
    for e in detail:
        titles = []
        for p in e["poles"]:
            lab = p.get("label")
            if lab and lab["confidence"] >= 0.5:
                titles.append(lab["text"])
                continue
            pick = [t for t in (p["measuredAgree"] + p["shared"] + p["inout"])]
            prim = next(r for r in p["reads"] if r["trust"] == "primary" and r["col"].startswith("read"))
            words = []
            for k in pick:
                disp = next((t[0] for r in p["reads"] for t in r["tokens"] if stem(t[0]) == k), None)
                if disp and disp.strip() and disp.strip() not in words:
                    words.append(disp.strip())
                if len(words) >= 3:
                    break
            if len(words) < 3 and L >= 13 and L <= 20 and p.get("peaks"):
                for t, _ in p["peaks"]:
                    if t.strip() and t.strip() not in words and not is_junk(tok.token_to_id(t) or 0):
                        words.append(t.strip())
                    if len(words) >= 3: break
            for t in prim["tokens"]:
                if len(words) >= 3: break
                s = t[0].strip()
                if s and not t[0].startswith("\u0001") and s not in words and not all(unicodedata.category(ch)[0] in "PSZC" for ch in s):
                    words.append(s)
            titles.append(" · ".join(words[:3]))
        e["titles"] = titles
        best = max(e["poles"], key=lambda p: p["score"])
        ref = len(neurons_index)
        neurons_index.append([L, e["n"], titles[0], titles[1], round(max(p["score"] for p in e["poles"]), 2),
                              (1 if any(p.get("label") for p in e["poles"]) else 0) | (2 if any(p.get("detector") for p in e["poles"]) else 0)
                              | (4 if any((p.get("shape") or {}).get("kind") == "predicts" for p in e["poles"]) else 0)
                              | (8 if any((p.get("shape") or {}).get("kind") == "suppresses" for p in e["poles"]) else 0)])
        for side, p in enumerate(e["poles"]):
            weights = collections.Counter()
            for r in p["reads"]:
                if r["trust"] not in ("primary", "trusted"):
                    continue
                for t in r["tokens"][:24]:
                    k = stem(t[0])
                    if not k or t[0].startswith("\u0001") or is_junk(tok.token_to_id(t[0]) or 0) and not k.isalnum():
                        continue
                    w = (3 if t[2] < 8 else 2 if t[2] < 24 else 1) + (2 if t[3] & 1 else 0) + (2 if t[3] & 2 else 0)
                    if weights[k] < w:
                        weights[k] = w
                        search_display.setdefault(k, t[0].strip())
            for t, c in p.get("peaks", []):
                k = stem(t)
                if k and k.strip():
                    weights[k] = max(weights[k], 4); search_display.setdefault(k, t.strip())
            lab = p.get("label")
            if lab:
                for word in re.findall(r"[\w']+", lab["text"].lower()):
                    if len(word) > 2:
                        weights[stem(word)] = max(weights[stem(word)], 5); search_display.setdefault(stem(word), word)
            for k, w in weights.most_common(40):
                search[k].append([ref, side, w])

    layers_meta.append({"layer": L, "band": band(L), "ins": ins, "outs": outs, "stats": stats, "hubs": hub_list[:16],
                        "count": len(detail), "missing": False,
                        "partial": [k for k, t in (("acts", acts), ("drivers", drivers), ("legible", legible)) if t is None]})
    for e in detail:
        for p in e["poles"]:
            for k in ("reads", "peaks", "drivers", "driversSum", "shared", "inout", "measuredAgree"):
                collect_cjk(p.get(k))
            for c in p.get("contexts", []):
                collect_cjk(c[1])
    json.dump({k.strip(): v for k, v in glossary.items() if v}, open(f"{out}/L{L:02d}.zh.json", "w"), ensure_ascii=False, separators=(",", ":"))
    glossary.clear()
    json.dump({"layer": L, "neurons": detail}, open(f"{out}/L{L:02d}.json", "w"), ensure_ascii=False, separators=(",", ":"))
    print(f"L{L:02d}: kept {len(detail)}  silent {stats and round(stats['silent'],2)} dark {stats and round(stats['dark'],2)}", flush=True)

# ---------- heads
heads = {}
hp = f"{catalog}/heads"
if os.path.isdir(hp):
    pat = pq.read_table(f"{hp}/patterns.parquet").to_pylist()
    cir = {(r["layer"], r["head"]): r for r in pq.read_table(f"{hp}/circuits.parquet").to_pylist()}
    cen = {(r["layer"], r["head"]): r for r in pq.read_table(f"{hp}/census.parquet").to_pylist()}
    for r in pat:
        k = (r["layer"], r["head"])
        c, s = cir.get(k, {}), cen.get(k, {})
        heads.setdefault(int(r["layer"]), []).append({
            "h": int(r["head"]), "kv": int(r["kv_head"]),
            "sink": round(r["sink"], 3), "prev": round(r["prev"], 3), "self": round(r["self"], 3),
            "induction": round(r["induction"], 3), "duplicate": round(r["duplicate"], 3),
            "copy": round(c.get("copy_S_R", 0) or 0, 3),
            "next": bool(s.get("next_mid")), "copies": bool(s.get("copy_mid")), "dark": bool(s.get("dark")),
            "reads": s.get("read_mid_z_str", "")})

for m in layers_meta:
    m["heads"] = sorted(heads.get(m["layer"], []), key=lambda h: h["h"])

json.dump({"model": "Qwen3-8B", "layers": LAYERS, "neuronsPerLayer": MLP, "total": LAYERS * MLP,
           "layerMeta": layers_meta, "neurons": neurons_index,
           "columns": ["layer", "neuron", "titlePlus", "titleMinus", "score", "flags"]},
          open(f"{out}/index.json", "w"), ensure_ascii=False, separators=(",", ":"))
json.dump({"forms": {k: v for k, v in search.items()}, "display": {k: search_display.get(k, k) for k in search}},
          open(f"{out}/search.json", "w"), ensure_ascii=False, separators=(",", ":"))
print("neurons", len(neurons_index), "search forms", len(search))
