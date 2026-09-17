import csv, collections, numpy as np
import exp

# distinct segment count per Uralic language
print("=== distinct segment inventory size per Uralic lang (target has 70 tokens) ===")
inv=[]
for lg in exp.URAL:
    segs={s for _,ss in exp.by_lang[lg] for s in ss}
    inv.append((len(segs),lg,exp.LANG_SUB[lg]))
for n,lg,sub in sorted(inv):
    print("  %3d  %s  %s"%(n,lg,sub))

# Load real test
test_rows=[]
with open('test.csv',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        test_rows.append((row['concept'],row['cipher'].split()))
print("\ntest rows",len(test_rows))

support=collections.defaultdict(list); seg_pool=set()
for lg in exp.URAL:
    for con,segs in exp.by_lang[lg]:
        support[con].append((lg,segs)); seg_pool.update(segs)

# run decipher with verbose to see coverage; patch to return langs coverage
tok_vocab=sorted({t for _,toks in test_rows for t in toks})
tok_id={t:i for i,t in enumerate(tok_vocab)}
seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
T=len(tok_vocab); S=len(seg_vocab)
cip_flat,cip_off,ref_flat,ref_off,pair_lang,langs,lang_idx=exp.build_pairs(test_rows,support,seg_id,tok_id)
nlang=len(langs); lang_w=np.ones(nlang)
C=np.ones((T,S))*0.01
for p in range(cip_off.shape[0]-1):
    ci=cip_flat[cip_off[p]:cip_off[p+1]]; rf=ref_flat[ref_off[p]:ref_off[p+1]]
    w=1.0/(len(ci)*len(rf))
    for t in ci: C[t,rf]+=w
sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
for it in range(10):
    counts,lcov,lcnt=exp.estep(cip_flat,cip_off,ref_flat,ref_off,pair_lang,lang_w,sub,-3.0,T,S,nlang)
    sub=np.log((counts+1e-6)/(counts+1e-6).sum(1,keepdims=True))
    cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0)
    mx=cov.max(); lang_w=(cov/mx)**3
print("\n=== REAL TEST: support-language coverage (higher=closer relative) ===")
order=np.argsort(-cov)
for i in order:
    print("  %s  sub=%-12s cov=%.3f"%(langs[i],exp.LANG_SUB[langs[i]],cov[i]))

from scipy.optimize import linear_sum_assignment
r,c=linear_sum_assignment(-counts)
tokmap={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
# confidence: for each token, counts mass on assigned seg / total
print("\n=== token map (by token frequency) with confidence ===")
tokfreq=collections.Counter(t for _,toks in test_rows for t in toks)
cnorm=counts/counts.sum(1,keepdims=True)
for t,fr in tokfreq.most_common(30):
    ti=tok_id[t]; sj=seg_id[tokmap[t]]
    print("  %s (freq %3d) -> '%s'  conf=%.2f"%(t,fr,tokmap[t],cnorm[ti,sj]))

# decode a few words
print("\n=== sample decoded words ===")
for con,toks in test_rows[:25]:
    print("  %-22s %s"%(con,' '.join(tokmap.get(t,'?') for t in toks)))
