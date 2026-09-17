import collections, random, numpy as np, sys, time
import exp, solve2
from scipy.optimize import linear_sum_assignment

def run_full(target, support_langs, wpow=12.0, gap=-3.0, n_iter=12):
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
    # inline EM to also get coverage
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(test_rows,support,seg_id,tok_id)
    nlang=len(langs); lw=np.ones(nlang)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lw,sub,gap,T,S,nlang,0.0)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        lw=(cov/mx)**wpow
    r,c=linear_sum_assignment(-(counts+1e-9))
    tokmap={tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}
    pred=[[tokmap.get(t,'?') for t in toks] for _,toks in test_rows]
    sc=exp.score_pred(pred,true_words)
    topcov=np.sort(cov)[::-1][:3]
    topl=[langs[i] for i in np.argsort(-cov)[:3]]
    return sc, topcov, topl

if __name__=='__main__':
    t0=time.time()
    print("target  sub          score   top3cov            top3langs")
    for tg in exp.URAL:
        sc,tc,tl=run_full(tg,exp.URAL)
        print("%-5s  %-11s  %.4f  [%s]  %s"%(tg,exp.LANG_SUB[tg],sc,
              ' '.join('%.3f'%x for x in tc),tl))
    print("(real target top-sister cov was ~0.708 -> pick analog)  [%.0fs]"%(time.time()-t0))
