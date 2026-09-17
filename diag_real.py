import csv, collections, numpy as np
import exp, solve2

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

tf=collections.Counter(t for _,toks in tr for t in toks)
tot=sum(tf.values())
# mass by rank
ranked=[t for t,_ in tf.most_common()]
for cut in [10,20,25,30,40,50,70]:
    m=sum(tf[t] for t in ranked[:cut])
    print("top-%d tokens carry %.1f%% of mass"%(cut,100*m/tot))
print("total tokens:",len(tok_vocab),"total seg-occ:",tot)

# per-concept: my decode vs Finnic sisters
FINN=['fin','ekk','krl','olo','vep']
bylc={(lg,con):segs for lg in FINN for con,segs in exp.by_lang[lg]}
print("\n=== decode vs Finnic sisters (sample) ===")
import random
rng=random.Random(0)
sample=rng.sample(test_rows,30)
for _id,con,toks in sample:
    dec=' '.join(tokmap.get(t,'?') for t in toks)
    sis=[]
    for lg in FINN:
        w=bylc.get((lg,con))
        if w: sis.append("%s:%s"%(lg,''.join(w)))
    print("%-20s MINE[%s]  %s"%(con[:20],dec,'  '.join(sis[:3])))
