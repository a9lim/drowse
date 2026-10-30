"""Chinese -> English glosses for displayed tokens, from CC-CEDICT (CC BY-SA 4.0, https://cc-cedict.org)."""
import re

TONES = {"a": "āáǎà", "e": "ēéěè", "i": "īíǐì", "o": "ōóǒò", "u": "ūúǔù", "ü": "ǖǘǚǜ"}
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def mark(syl):
    """'hao3' -> 'hǎo'."""
    m = re.match(r"^([a-zü:]+)([1-5])$", syl.lower().replace("u:", "ü"))
    if not m:
        return syl.replace("u:", "ü")
    s, t = m.group(1), int(m.group(2))
    if t == 5:
        return s
    for v in ("a", "e"):
        if v in s:
            return s.replace(v, TONES[v][t - 1], 1)
    if "ou" in s:
        return s.replace("o", TONES["o"][t - 1], 1)
    for i in range(len(s) - 1, -1, -1):
        if s[i] in TONES:
            return s[:i] + TONES[s[i]][t - 1] + s[i + 1:]
    return s


def load(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.startswith("#"):
            continue
        m = re.match(r"^(\S+) (\S+) \[([^\]]*)\] /(.*)/$", line.strip())
        if not m:
            continue
        trad, simp, py, defs = m.groups()
        defs = re.sub(r"\s*[\u3400-\u9fff|]+\[[^\]]*\]", "", defs)
        senses = [x for x in defs.split("/") if x and not x.startswith(("variant of", "old variant", "CL:", "used in", "surname "))]
        if not senses:
            senses = [x for x in defs.split("/") if x][:1]
        proper = py[:1].isupper()
        entry = (" ".join(mark(s) for s in py.split()), senses, proper)
        for k in (simp, trad):
            # Keep the first, usually most common, reading; a capitalised pinyin marks a proper noun, so prefer a common noun.
            if k not in d or (d[k][2] and not proper):
                d[k] = entry
    return d


def gloss(text, d, max_len=6):
    """Longest-match segmentation over the CJK run(s) in text: (pinyin, meaning) or None."""
    s = text.strip()
    if not CJK.search(s):
        return None
    if s in d:
        py, senses, _ = d[s]
        return py, "; ".join(senses[:3])
    parts, i = [], 0
    while i < len(s):
        for j in range(min(len(s), i + max_len), i, -1):
            if s[i:j] in d:
                parts.append((s[i:j], d[s[i:j]]))
                i = j
                break
        else:
            if CJK.match(s[i]):
                return None
            i += 1
    if not parts:
        return None
    py = " ".join(p[1][0] for p in parts)
    short = lambda senses: re.sub(r"\s*\(.*?\)", "", senses[0]).split(";")[0].strip() or senses[0]
    meaning = " + ".join(f"{w} ({short(p[1])})" for w, p in parts) if len(parts) > 1 else "; ".join(parts[0][1][1][:3])
    return py, meaning
