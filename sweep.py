import collections, random, numpy as np, sys, time
import exp, solve2

FINNIC=['fin','ekk','krl','olo','vep']

def run_target(target, support_langs, **kw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    seg2tok={s:"x%d"%i for i,s in enumerate(perm)}
    test_rows=[]; true_words=[]
    for con,segs in words:
        test_rows.append((con,[seg2tok[s] for s in segs])); true_words.append(segs)
    support=collections.defaultdict(list); seg_pool=set()
    for lg in support_langs:
        if lg==target: continue
        for con,segs in exp.by_lang[lg]:
            support[con].append((lg,segs)); seg_pool.update(segs)
    tokmap=solve2.decipher2(test_rows,support,seg_pool,**kw)
    pred=[[tokmap.get(t,'?') for t in toks] for _,toks in test_rows]
    return exp.score_pred(pred,true_words)

def evalcfg(name, support_mode, **kw):
    scs=[]
    for tg in FINNIC:
        if support_mode=='uralic': sl=exp.URAL
        elif support_mode=='finnic': sl=FINNIC
        scs.append(run_target(tg,sl,**kw))
    print("%-22s ekk=%.3f finnicMEAN=%.4f  per:%s"%(name,scs[1],np.mean(scs),
          ' '.join('%s=%.3f'%(t,s) for t,s in zip(FINNIC,scs))))
    return scs

if __name__=='__main__':
    t0=time.time()
    base=dict(n_iter=10,gap=-3.0,relpow=0.0,smooth=False)
    print("--- wpow sweep (uralic support) ---")
    for wp in [3.0,5.0,8.0,12.0,20.0]:
        evalcfg("wpow=%g"%wp,'uralic',wpow=wp,**base)
    print("--- gap sweep (uralic, wpow=8) ---")
    for gp in [-2.0,-3.0,-4.0,-5.0]:
        evalcfg("gap=%g"%gp,'uralic',wpow=8.0,n_iter=10,relpow=0.0,smooth=False)
    print("--- finnic-only support ---")
    for wp in [3.0,8.0]:
        evalcfg("finnic wpow=%g"%wp,'finnic',wpow=wp,n_iter=10,gap=-3.0,relpow=0.0,smooth=False)
    print("--- smooth + wpow=8 (uralic) ---")
    evalcfg("smooth wpow=8",'uralic',wpow=8.0,n_iter=10,gap=-3.0,relpow=0.0,smooth=True)
    print("--- n_iter sweep (uralic wpow=8) ---")
    for ni in [6,15,25]:
        evalcfg("niter=%d"%ni,'uralic',wpow=8.0,n_iter=ni,gap=-3.0,relpow=0.0,smooth=False)
    print("total %.0fs"%(time.time()-t0))
