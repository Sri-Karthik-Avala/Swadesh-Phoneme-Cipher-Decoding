import csv, collections, numpy as np
import exp, solve2, phon

test_rows=[]
with open('test.csv',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        test_rows.append((row['id'],row['concept'],row['cipher'].split()))
tr=[(c,toks) for _id,c,toks in test_rows]
support=collections.defaultdict(list); seg_pool=set()
for lg in exp.URAL:
    for con,segs in exp.by_lang[lg]:
        support[con].append((lg,segs)); seg_pool.update(segs)
tokmap,tok_vocab,tok_id,seg_vocab,counts=solve2.decipher2(
    tr,support,seg_pool,n_iter=12,wpow=12.0,gap=-3.0,relpow=0.5,smooth=False,return_counts=True)

# classify each seg as vowel/consonant via featurizer
def is_vowel(s):
    v=phon.feat_vec(s)
    return v[4]>0.5  # is-vowel flag
segvow=np.array([is_vowel(s) for s in seg_vocab])
cn=counts/counts.sum(1,keepdims=True)
tf=collections.Counter(t for _,toks in tr for t in toks)

print("token freq  pred   vmass  top3-candidates")
for t,fr in tf.most_common(40):
    ti=tok_id[t]
    row=counts[ti]
    vmass=row[segvow].sum()/row.sum()
    top=np.argsort(-row)[:3]
    cand=' '.join("%s:%.2f"%(seg_vocab[j],cn[ti,j]) for j in top)
    predv='V' if is_vowel(tokmap[t]) else 'C'
    print("  %-4s %4d  %-4s%s v=%.2f  %s"%(t,fr,tokmap[t],predv,vmass,cand))
