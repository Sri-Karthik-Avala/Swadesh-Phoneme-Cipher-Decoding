# made by - Karthik
import sys, os, csv, collections
from pathlib import Path
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        def deco(fn): return fn
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return deco

from scipy.optimize import linear_sum_assignment

EM_ITERS = int(os.environ.get("EM_ITERS", "12"))
WPOW = float(os.environ.get("WPOW", "12.0"))
RELPOW = float(os.environ.get("RELPOW", "0.5"))
GAP = float(os.environ.get("GAP", "-3.0"))
ANCHOR_ITERS = int(os.environ.get("ANCHOR_ITERS", "6"))
SKELPOW = float(os.environ.get("SKELPOW", "6.0"))
MARG = float(os.environ.get("MARG", "0.0"))
CATCONSTRAIN = int(os.environ.get("CATCONSTRAIN", "1"))
TARGET_FAMILY = os.environ.get("TARGET_FAMILY", "Uralic")

VOWEL_CHARS = set("aeiouɑɐæɛɜəɘɤɨɪɯoɔøœɒʉʊʌyʏɵ")

DIACRITICS = ["ː", "ˑ", "ʲ", "ʷ", "ʰ", "̥", "̊", "ʼ", "́", "̀", "̄", "̂", "̃", "̈", "̪", "̻", "̠", "ⁿ"]

def strip_diacritics(seg):
    base = seg
    for d in DIACRITICS:
        base = base.replace(d, "")
    return base if base else seg

def is_vowel_seg(seg):
    base = strip_diacritics(seg)
    if not base:
        return False
    vc = sum(1 for c in base if c in VOWEL_CHARS)
    cc = sum(1 for c in base if c.isalpha() and c not in VOWEL_CHARS)
    return vc >= cc and vc > 0

def find_data_root(argv):
    if len(argv) > 1 and argv[1]:
        p = Path(argv[1])
        if (p / "test.csv").exists():
            return p
    for c in [Path("."), Path("dataset/public"), Path("../dataset/public"), Path("data")]:
        if (c / "test.csv").exists():
            return c
    for base in [Path("."), Path(".."), Path("../..")]:
        for cand in base.rglob("test.csv"):
            if (cand.parent / "train.csv").exists():
                return cand.parent
    return Path(".")

def load_train(path):
    lang_fam = {}
    by_lang = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lg = row["language"]; lang_fam[lg] = row["family"]
            by_lang[lg].append((row["concept"], row["ipa"].split()))
    return lang_fam, by_lang

def load_test(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append((row["id"], row["concept"], row["cipher"].split()))
    return rows

@njit(cache=True)
def estep(cip_flat, cip_off, ref_flat, ref_off, pair_lang, lang_w, sub_score, gap, T, S, nlang, relpow):
    counts = np.zeros((T, S))
    lang_cov = np.zeros(nlang); lang_cnt = np.zeros(nlang)
    npair = cip_off.shape[0] - 1
    MAXN = 40; MAXM = 60
    D = np.empty((MAXN + 1, MAXM + 1)); B = np.empty((MAXN + 1, MAXM + 1), dtype=np.int8)
    mt = np.empty(MAXN + MAXM, dtype=np.int64); ms = np.empty(MAXN + MAXM, dtype=np.int64)
    for p in range(npair):
        lg = pair_lang[p]; w = lang_w[lg]
        cs = cip_off[p]; ce = cip_off[p + 1]; n = ce - cs
        rs = ref_off[p]; re = ref_off[p + 1]; m = re - rs
        if n > MAXN or m > MAXM or n == 0 or m == 0:
            continue
        D[0, 0] = 0.0
        for i in range(1, n + 1): D[i, 0] = D[i - 1, 0] + gap
        for j in range(1, m + 1): D[0, j] = D[0, j - 1] + gap
        for i in range(1, n + 1):
            ti = cip_flat[cs + i - 1]
            for j in range(1, m + 1):
                sj = ref_flat[rs + j - 1]
                diag = D[i - 1, j - 1] + sub_score[ti, sj]
                up = D[i - 1, j] + gap; left = D[i, j - 1] + gap
                if diag >= up and diag >= left:
                    D[i, j] = diag; B[i, j] = 0
                elif up >= left:
                    D[i, j] = up; B[i, j] = 1
                else:
                    D[i, j] = left; B[i, j] = 2
        i = n; j = m; matched = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and B[i, j] == 0:
                mt[matched] = cip_flat[cs + i - 1]; ms[matched] = ref_flat[rs + j - 1]
                matched += 1; i -= 1; j -= 1
            elif i > 0 and (j == 0 or B[i, j] == 1):
                i -= 1
            else:
                j -= 1
        mx = n if n > m else m
        rel = matched / mx
        pw = w * (rel ** relpow)
        for a in range(matched):
            counts[mt[a], ms[a]] += pw
        lang_cov[lg] += rel; lang_cnt[lg] += 1.0
    return counts, lang_cov, lang_cnt

@njit(cache=True)
def estep_pin(cip_flat, cip_off, ref_flat, ref_off, pair_lang, lang_w, sub_score, gap, T, S, nlang,
              tokpred_base, tok_isvow, seg_base, skelpow):
    counts = np.zeros((T, S))
    lang_cov = np.zeros(nlang); lang_cnt = np.zeros(nlang)
    npair = cip_off.shape[0] - 1
    MAXN = 40; MAXM = 60
    D = np.empty((MAXN + 1, MAXM + 1)); B = np.empty((MAXN + 1, MAXM + 1), dtype=np.int8)
    mt = np.empty(MAXN + MAXM, dtype=np.int64); ms = np.empty(MAXN + MAXM, dtype=np.int64)
    for p in range(npair):
        lg = pair_lang[p]; w = lang_w[lg]
        cs = cip_off[p]; ce = cip_off[p + 1]; n = ce - cs
        rs = ref_off[p]; re = ref_off[p + 1]; m = re - rs
        if n > MAXN or m > MAXM or n == 0 or m == 0:
            continue
        D[0, 0] = 0.0
        for i in range(1, n + 1): D[i, 0] = D[i - 1, 0] + gap
        for j in range(1, m + 1): D[0, j] = D[0, j - 1] + gap
        for i in range(1, n + 1):
            ti = cip_flat[cs + i - 1]
            for j in range(1, m + 1):
                sj = ref_flat[rs + j - 1]
                diag = D[i - 1, j - 1] + sub_score[ti, sj]
                up = D[i - 1, j] + gap; left = D[i, j - 1] + gap
                if diag >= up and diag >= left:
                    D[i, j] = diag; B[i, j] = 0
                elif up >= left:
                    D[i, j] = up; B[i, j] = 1
                else:
                    D[i, j] = left; B[i, j] = 2
        i = n; j = m; matched = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and B[i, j] == 0:
                mt[matched] = cip_flat[cs + i - 1]; ms[matched] = ref_flat[rs + j - 1]
                matched += 1; i -= 1; j -= 1
            elif i > 0 and (j == 0 or B[i, j] == 1):
                i -= 1
            else:
                j -= 1
        ncons_word = 0
        for i2 in range(n):
            if tok_isvow[cip_flat[cs + i2]] == 0: ncons_word += 1
        cmatch = 0
        for a in range(matched):
            t = mt[a]
            if tok_isvow[t] == 0 and seg_base[ms[a]] == tokpred_base[t]: cmatch += 1
        denom = ncons_word if ncons_word > 0 else 1
        skel = cmatch / denom
        pw = w * (skel ** skelpow)
        for a in range(matched):
            counts[mt[a], ms[a]] += pw
        mx = n if n > m else m
        lang_cov[lg] += matched / mx; lang_cnt[lg] += 1.0
    return counts, lang_cov, lang_cnt

def build_pairs(test_rows, support_by_concept, seg_id, tok_id):
    langs_set = set(); plist = []
    for _id, con, toks in test_rows:
        ci = [tok_id[t] for t in toks]
        for lg, segs in support_by_concept.get(con, []):
            rf = [seg_id[s] for s in segs if s in seg_id]
            if rf:
                plist.append((ci, rf, lg)); langs_set.add(lg)
    langs = sorted(langs_set); lang_idx = {l: i for i, l in enumerate(langs)}
    cip_flat = []; cip_off = [0]; ref_flat = []; ref_off = [0]; pair_lang = []
    for ci, rf, lg in plist:
        cip_flat.extend(ci); cip_off.append(len(cip_flat))
        ref_flat.extend(rf); ref_off.append(len(ref_flat))
        pair_lang.append(lang_idx[lg])
    return (np.array(cip_flat, np.int32), np.array(cip_off, np.int64),
            np.array(ref_flat, np.int32), np.array(ref_off, np.int64),
            np.array(pair_lang, np.int64), langs)

def category_assign(score, tok_isvow, seg_isvow, tok_vocab, seg_vocab):
    T, S = score.shape
    tokmap = {}
    for cat in [0, 1]:
        tks = [i for i in range(T) if tok_isvow[i] == cat]
        sgs = [j for j in range(S) if seg_isvow[j] == cat]
        if not tks:
            continue
        if not sgs:
            sgs = list(range(S))
        sub = score[np.ix_(tks, sgs)]
        r, c = linear_sum_assignment(-sub)
        assigned = set()
        for ri, ci in zip(r, c):
            tokmap[tok_vocab[tks[ri]]] = seg_vocab[sgs[ci]]; assigned.add(ri)
        for ri in range(len(tks)):
            if ri not in assigned:
                j = int(np.argmax(score[tks[ri]]))
                tokmap[tok_vocab[tks[ri]]] = seg_vocab[j]
    return tokmap

def decipher(test_rows, support_by_concept, seg_pool):
    tok_vocab = sorted({t for _id, _c, toks in test_rows for t in toks})
    tok_id = {t: i for i, t in enumerate(tok_vocab)}
    seg_vocab = sorted(seg_pool); seg_id = {s: i for i, s in enumerate(seg_vocab)}
    T = len(tok_vocab); S = len(seg_vocab)
    seg_isvow = np.array([1 if is_vowel_seg(s) else 0 for s in seg_vocab], dtype=np.int64)
    bases = sorted(set(strip_diacritics(s) for s in seg_vocab)); base_id = {b: i for i, b in enumerate(bases)}
    seg_base = np.array([base_id[strip_diacritics(s)] for s in seg_vocab], dtype=np.int64)
    cf, co, rf, ro, pl, langs = build_pairs(test_rows, support_by_concept, seg_id, tok_id)
    nlang = len(langs); lang_w = np.ones(nlang)
    C = np.ones((T, S)) * 0.01
    for p in range(co.shape[0] - 1):
        ci = cf[co[p]:co[p + 1]]; rr = rf[ro[p]:ro[p + 1]]
        w = 1.0 / (len(ci) * len(rr))
        for t in ci:
            C[t, rr] += w
    sub = np.log(C / C.sum(1, keepdims=True) + 1e-9)
    counts = None
    for it in range(EM_ITERS):
        counts, lcov, lcnt = estep(cf, co, rf, ro, pl, lang_w, sub, GAP, T, S, nlang, RELPOW)
        cs = counts + 1e-6
        sub = np.log(cs / cs.sum(1, keepdims=True))
        cov = np.where(lcnt > 0, lcov / np.maximum(lcnt, 1), 0.0)
        mx = cov.max() if cov.size else 1.0
        if mx > 0:
            lang_w = (cov / mx) ** WPOW
    tok_isvow = ((counts * seg_isvow[None, :]).sum(1) / (counts.sum(1) + 1e-9) > 0.5).astype(np.int64)
    for it in range(ANCHOR_ITERS):
        tokpred = np.zeros(T, dtype=np.int64)
        for t in range(T):
            row = counts[t].copy(); mask = (seg_isvow == tok_isvow[t])
            if mask.any():
                row = np.where(mask, row, -1.0)
            tokpred[t] = int(np.argmax(row))
        tokpred_base = seg_base[tokpred]
        counts, lcov, lcnt = estep_pin(cf, co, rf, ro, pl, lang_w, sub, GAP, T, S, nlang, tokpred_base, tok_isvow, seg_base, SKELPOW)
        cs = counts + 1e-6
        sub = np.log(cs / cs.sum(1, keepdims=True))
        cov = np.where(lcnt > 0, lcov / np.maximum(lcnt, 1), 0.0)
        mx = cov.max() if cov.size else 1.0
        if mx > 0:
            lang_w = (cov / mx) ** WPOW
    marg = counts.sum(0) + 1e-12
    score = (counts + 1e-12) / (marg[None, :] ** MARG)
    if CATCONSTRAIN:
        tokmap = category_assign(score, tok_isvow, seg_isvow, tok_vocab, seg_vocab)
    else:
        r, c = linear_sum_assignment(-score)
        tokmap = {tok_vocab[ri]: seg_vocab[ci] for ri, ci in zip(r, c)}
    assigned = score.argmax(1)
    for i, t in enumerate(tok_vocab):
        if t not in tokmap:
            tokmap[t] = seg_vocab[int(assigned[i])]
    return tokmap

def main():
    argv = sys.argv
    root = find_data_root(argv)
    out_path = Path(argv[2]) if len(argv) > 2 and argv[2] else Path("working/submission.csv")
    lang_fam, by_lang = load_train(root / "train.csv")
    test_rows = load_test(root / "test.csv")
    support_langs = [l for l in lang_fam if lang_fam[l] == TARGET_FAMILY]
    if len(support_langs) < 3:
        support_langs = list(lang_fam)
    support = collections.defaultdict(list); seg_pool = set()
    for lg in support_langs:
        for con, segs in by_lang[lg]:
            support[con].append((lg, segs)); seg_pool.update(segs)
    print("langs=%d support=%d segpool=%d testrows=%d anchor=%d marg=%.2f cat=%d" % (
        len(lang_fam), len(support_langs), len(seg_pool), len(test_rows), ANCHOR_ITERS, MARG, CATCONSTRAIN))
    tokmap = decipher(test_rows, support, seg_pool)
    print("tokens mapped:", len(tokmap))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["id", "ipa"])
        for _id, con, toks in test_rows:
            wtr.writerow([_id, " ".join(tokmap.get(t, "a") for t in toks)])
    print("wrote", out_path)

if __name__ == "__main__":
    main()
