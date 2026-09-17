import csv, collections, numpy as np
import exp, solve2, solve4, phon

test_rows=[]
with open('test.csv',encoding='utf-8') as f:
    for row in csv.DictReader(f):
        test_rows.append((row['id'],row['concept'],row['cipher'].split()))
tr=[(c,toks) for _id,c,toks in test_rows]
support=collections.defaultdict(list); seg_pool=set()
for lg in exp.URAL:
    for con,segs in exp.by_lang[lg]:
        support[con].append((lg,segs)); seg_pool.update(segs)

FINN=['fin','ekk','krl','olo','vep']
bylc={(lg,con):segs for lg in FINN for con,segs in exp.by_lang[lg]}
known={'1_eye':'silmæ','197_girl':'','23_water':'','380_ring':'sormus'}

def decode_and_show(name, emkw, akw):
    E=solve4.em_counts(tr,support,seg_pool,**emkw)
    tm=solve4.assign(E['counts'],E['tok_isvow'],E['seg_isvow'],E['tok_vocab'],E['seg_vocab'],**akw)
    tf=collections.Counter(t for _,toks in tr for t in toks)
    print("=== %s ===  top-14 token map:"%name)
    line=' '.join("%s->%s"%(t,tm[t]) for t,_ in tf.most_common(14))
    print("  "+line)
    probes=['1_eye','380_ring','23_water','736_drink','484_short','875_pull','33_toe','494_hard','822_put','146_mouse','523_dark']
    for con in probes:
        for _id,c,toks in test_rows:
            if c==con:
                dec=' '.join(tm.get(t,'?') for t in toks)
                sis=' '.join("%s:%s"%(lg,''.join(bylc.get((lg,con),[]))) for lg in FINN if bylc.get((lg,con)))
                print("  %-14s [%s]  | %s"%(con,dec,sis))
                break
    return tm

emk_v1=dict(n_iter=12,wpow=12.0,gap=-3.0,relpow=0.5,anchor_iters=0,skelpow=3.0)
emk_anc=dict(n_iter=12,wpow=12.0,gap=-3.0,relpow=0.5,anchor_iters=6,skelpow=3.0)
decode_and_show("v1 (raw,noCat)",emk_v1,dict(marg_alpha=0.0,catconstrain=False))
print()
decode_and_show("v2 anchor+marg0.5+cat",emk_anc,dict(marg_alpha=0.5,catconstrain=True))
