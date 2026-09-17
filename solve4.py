import csv, collections, random, sys, time
import numpy as np
from scipy.optimize import linear_sum_assignment
import exp, solve2, solve3, phon

def em_counts(test_rows, support_by_concept, seg_pool, n_iter=12, wpow=12.0, gap=-3.0,
              relpow=0.5, anchor_iters=6, skelpow=3.0):
    tok_vocab=sorted({t for _,toks in test_rows for t in toks})
    tok_id={t:i for i,t in enumerate(tok_vocab)}
    seg_vocab=sorted(seg_pool); seg_id={s:i for i,s in enumerate(seg_vocab)}
    T=len(tok_vocab); S=len(seg_vocab)
    cf,co,rf,ro,pl,langs,li=exp.build_pairs(test_rows,support_by_concept,seg_id,tok_id)
    nlang=len(langs); lang_w=np.ones(nlang)
    seg_isvow=np.array([1 if phon.feat_vec(s)[4]>0.5 else 0 for s in seg_vocab],dtype=np.int64)
    C=np.ones((T,S))*0.01
    for p in range(co.shape[0]-1):
        ci=cf[co[p]:co[p+1]]; rr=rf[ro[p]:ro[p+1]]; w=1.0/(len(ci)*len(rr))
        for t in ci: C[t,rr]+=w
    sub=np.log(C/C.sum(1,keepdims=True)+1e-9)
    counts=None
    for it in range(n_iter):
        counts,lcov,lcnt=solve2.estep2(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,relpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    vmass=(counts*seg_isvow[None,:]).sum(1)/(counts.sum(1)+1e-9)
    tok_isvow=(vmass>0.5).astype(np.int64)
    for it in range(anchor_iters):
        tokpred=np.zeros(T,dtype=np.int64)
        for t in range(T):
            row=counts[t].copy(); mask=(seg_isvow==tok_isvow[t])
            if mask.any(): row=np.where(mask,row,-1.0)
            tokpred[t]=int(np.argmax(row))
        counts,lcov,lcnt=solve3.estep_anchored(cf,co,rf,ro,pl,lang_w,sub,gap,T,S,nlang,tokpred,tok_isvow,skelpow)
        cs=counts+1e-6; sub=np.log(cs/cs.sum(1,keepdims=True))
        cov=np.where(lcnt>0,lcov/np.maximum(lcnt,1),0.0); mx=cov.max()
        if mx>0: lang_w=(cov/mx)**wpow
    return dict(counts=counts,tok_vocab=tok_vocab,seg_vocab=seg_vocab,
                tok_isvow=tok_isvow,seg_isvow=seg_isvow)

def assign(counts, tok_isvow, seg_isvow, tok_vocab, seg_vocab, marg_alpha=0.0, catconstrain=True):
    T,S=counts.shape
    marg=counts.sum(0)+1e-9
    score=(counts+1e-9)/(marg[None,:]**marg_alpha)
    if catconstrain:
        return solve3.category_assign(score,tok_isvow,seg_isvow,tok_vocab,seg_vocab)
    r,c=linear_sum_assignment(-score)
    return {tok_vocab[ri]:seg_vocab[ci] for ri,ci in zip(r,c)}

def run(target, emkw, **akw):
    words=exp.by_lang[target]
    segset=sorted({s for _,segs in words for s in segs})
    rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
    s2t={s:"x%d"%i for i,s in enumerate(perm)}
    tr=[(con,[s2t[s] for s in segs]) for con,segs in words]
    true=[segs for con,segs in words]
    support,seg_pool=solve2.make_support(target)
    E=em_counts(tr,support,seg_pool,**emkw)
    tm=assign(E['counts'],E['tok_isvow'],E['seg_isvow'],E['tok_vocab'],E['seg_vocab'],**akw)
    pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
    return exp.score_pred(pred,true), E

if __name__=='__main__':
    TARG=['fin','ekk','krl','olo','vep','sme','smj','sjd','sms','myv','udm']
    t0=time.time()
    # precompute EM once per target for each anchor setting, then sweep marg_alpha
    for anchor in [0,6]:
        emkw=dict(n_iter=12,wpow=12.0,gap=-3.0,relpow=0.5,anchor_iters=anchor,skelpow=3.0)
        Es={}
        for tg in TARG:
            _,E=run(tg,emkw,marg_alpha=0.0)
            Es[tg]=E
        for ma in [0.0,0.5,0.75,1.0,1.25]:
            scs=[]
            for tg in TARG:
                E=Es[tg]
                tm=assign(E['counts'],E['tok_isvow'],E['seg_isvow'],E['tok_vocab'],E['seg_vocab'],marg_alpha=ma,catconstrain=True)
                words=exp.by_lang[tg]; segset=sorted({s for _,segs in words for s in segs})
                rng=random.Random(12345); perm=segset[:]; rng.shuffle(perm)
                s2t={s:"x%d"%i for i,s in enumerate(perm)}
                tr=[(con,[s2t[s] for s in segs]) for con,segs in words]; true=[segs for con,segs in words]
                pred=[[tm.get(t,'?') for t in toks] for _,toks in tr]
                scs.append(exp.score_pred(pred,true))
            print("anchor=%d marg=%.2f | finnic=%.4f hard=%.4f all=%.4f | %s"%(anchor,ma,
                np.mean(scs[:5]),np.mean(scs[5:9]),np.mean(scs),
                ' '.join('%s=%.3f'%(t,s) for t,s in zip(TARG,scs))))
    print("t=%.0fs"%(time.time()-t0))
